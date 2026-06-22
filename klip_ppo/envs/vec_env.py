"""Vectorised rollout collector implementing the ``Collector`` protocol."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from klip_ppo.configs.env import EnvConfig
from klip_ppo.configs.rollout import RolloutConfig
from klip_ppo.core.distributions import PolicyDistParams
from klip_ppo.core.gae import compute_gae
from klip_ppo.core.networks import ActorCritic
from klip_ppo.core.rollout import EpisodeStats, RolloutBatch
from klip_ppo.envs.gym_env import make_env
from klip_ppo.envs.normalization import EnvNormalizer


class VectorCollector:
    """
    Collect on-policy rollouts from a gymnasium vector env.

    Side-effects worth knowing:

    - Stores the policy's distribution parameters at each rollout step
      (logits for categorical envs; mean+log_std for diagonal-Gaussian
      envs) so that the loss strategies can compute the full
      ``KL(π_old || π_new)`` against the post-update policy.
    - Records both *raw* (pre-normalisation) and *wrapped* (what the
      trainer saw) episode returns. ``RecordEpisodeStatistics`` is
      injected near the base env by ``make_env``, so the raw stream is
      populated whenever an episode terminates.
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        rollout_cfg: RolloutConfig,
        *,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
        seed: int,
    ) -> None:
        env_fns = [
            make_env(env_cfg, seed=seed, env_idx=i) for i in range(rollout_cfg.num_envs)
        ]
        self.envs: gym.vector.VectorEnv
        if rollout_cfg.async_envs and rollout_cfg.num_envs > 1:
            self.envs = gym.vector.AsyncVectorEnv(
                env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
            )
        else:
            self.envs = gym.vector.SyncVectorEnv(
                env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
            )
        self._num_envs = int(rollout_cfg.num_envs)
        self._n_steps = int(rollout_cfg.n_steps)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._action_is_discrete = isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        )

        obs: object
        obs, _ = self.envs.reset(seed=seed)
        raw_obs = np.asarray(obs, dtype=np.float32)
        self.normalizer = EnvNormalizer(
            obs_shape=tuple(int(x) for x in raw_obs.shape[1:]),
            num_envs=self._num_envs,
            normalize_obs=env_cfg.normalize_obs,
            normalize_reward=env_cfg.normalize_reward,
            clip_obs=env_cfg.clip_obs,
            clip_reward=env_cfg.clip_reward,
            reward_scale=env_cfg.reward_scale,
            gamma=gamma,
        )
        self._obs = self.normalizer.normalize_obs(raw_obs, update=True)
        self._ep_wrapped_return = np.zeros(self._num_envs, dtype=np.float64)
        self._ep_len = np.zeros(self._num_envs, dtype=np.int64)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def close(self) -> None:
        self.envs.close()

    def state_dict(self) -> dict[str, Any]:
        return {"normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "normalizer" in state:
            self.normalizer.load_state_dict(
                state["normalizer"], num_envs=self._num_envs
            )

    def _value_truncated_final_obs(
        self,
        *,
        policy: ActorCritic,
        info: Any,
        truncated: np.ndarray,
    ) -> torch.Tensor:
        out = torch.zeros(self._num_envs, dtype=torch.float32, device=self.device)
        if not bool(np.any(truncated)):
            return out

        final_obs = _extract_final_observations(info, self._num_envs)
        if final_obs is None:
            return out

        indices: list[int] = []
        obs_rows: list[np.ndarray] = []
        for i in range(self._num_envs):
            if truncated[i] and final_obs[i] is not None:
                indices.append(i)
                obs_rows.append(np.asarray(final_obs[i], dtype=np.float32))
        if not obs_rows:
            return out

        obs_arr = np.stack(obs_rows, axis=0)
        obs_arr = self.normalizer.normalize_obs(obs_arr, update=False)
        obs_t = torch.as_tensor(obs_arr, device=self.device, dtype=torch.float32)
        values = policy.value(obs_t)
        out[torch.as_tensor(indices, device=self.device)] = values
        return out

    def collect(self, policy: ActorCritic) -> tuple[RolloutBatch, EpisodeStats]:
        T, E = self._n_steps, self._num_envs
        obs_shape = self._obs.shape[1:]
        device = self.device

        obs_buf = torch.zeros((T, E, *obs_shape), dtype=torch.float32, device=device)
        rewards_buf = torch.zeros((T, E), dtype=torch.float32, device=device)
        terminated_buf = torch.zeros((T, E), dtype=torch.bool, device=device)
        truncated_buf = torch.zeros((T, E), dtype=torch.bool, device=device)
        truncated_value_buf = torch.zeros((T, E), dtype=torch.float32, device=device)
        values_buf = torch.zeros((T, E), dtype=torch.float32, device=device)
        logp_buf = torch.zeros((T, E), dtype=torch.float32, device=device)
        if self._action_is_discrete:
            actions_buf = torch.zeros((T, E), dtype=torch.int64, device=device)
        else:
            action_space = self.envs.single_action_space
            if not isinstance(action_space, gym.spaces.Box):
                raise NotImplementedError(
                    "only discrete and flat Box action spaces are supported"
                )
            action_dim = int(action_space.shape[0])
            actions_buf = torch.zeros(
                (T, E, action_dim), dtype=torch.float32, device=device
            )

        # Distribution-parameter snapshots, lazily allocated once we see
        # the first action's params shape. Categorical: (T, E, n_actions).
        # Diag Gaussian: (T, E, action_dim) for both mean and log_std.
        logits_buf: torch.Tensor | None = None
        mean_buf: torch.Tensor | None = None
        log_std_buf: torch.Tensor | None = None

        ep_raw_returns: list[float] = []
        ep_wrapped_returns: list[float] = []
        ep_lengths: list[float] = []

        was_training = policy.training
        policy.eval()
        try:
            for t in range(T):
                obs_t = torch.as_tensor(self._obs, device=device, dtype=torch.float32)
                action, logprob, value, params = policy.act(obs_t)

                if params.kind == "categorical":
                    assert params.logits is not None
                    if logits_buf is None:
                        logits_buf = torch.zeros(
                            (T, *params.logits.shape),
                            dtype=params.logits.dtype,
                            device=device,
                        )
                    logits_buf[t] = params.logits
                else:
                    assert params.mean is not None and params.log_std is not None
                    if mean_buf is None:
                        mean_buf = torch.zeros(
                            (T, *params.mean.shape),
                            dtype=params.mean.dtype,
                            device=device,
                        )
                        log_std_buf = torch.zeros_like(mean_buf)
                    mean_buf[t] = params.mean
                    assert log_std_buf is not None
                    log_std_buf[t] = params.log_std

                if self._action_is_discrete:
                    np_action = action.detach().cpu().numpy().astype(np.int64)
                else:
                    np_action = action.detach().cpu().numpy().astype(np.float32)

                next_obs, raw_reward, terminated, truncated, info = self.envs.step(
                    np_action
                )
                terminated = np.asarray(terminated, dtype=bool)
                truncated = np.asarray(truncated, dtype=bool)
                done = np.logical_or(terminated, truncated)
                reward = self.normalizer.normalize_reward(
                    np.asarray(raw_reward, dtype=np.float32),
                    terminated=terminated,
                    truncated=truncated,
                    update=True,
                )

                obs_buf[t] = obs_t
                actions_buf[t] = action
                logp_buf[t] = logprob
                values_buf[t] = value
                rewards_buf[t] = torch.as_tensor(
                    reward, device=device, dtype=torch.float32
                )
                terminated_buf[t] = torch.as_tensor(
                    terminated, device=device, dtype=torch.bool
                )
                truncated_buf[t] = torch.as_tensor(
                    truncated, device=device, dtype=torch.bool
                )
                truncated_value_buf[t] = self._value_truncated_final_obs(
                    policy=policy,
                    info=info,
                    truncated=truncated,
                )

                self._ep_wrapped_return += np.asarray(reward, dtype=np.float64)
                self._ep_len += 1

                raw_per_env = _extract_raw_returns(info, E)
                for i in range(E):
                    if done[i]:
                        ep_wrapped_returns.append(float(self._ep_wrapped_return[i]))
                        ep_lengths.append(float(self._ep_len[i]))
                        self._ep_wrapped_return[i] = 0.0
                        self._ep_len[i] = 0
                        raw_return = raw_per_env[i] if raw_per_env is not None else None
                        if raw_return is not None:
                            ep_raw_returns.append(float(raw_return))

                self._obs = self.normalizer.normalize_obs(
                    np.asarray(next_obs, dtype=np.float32), update=True
                )

            final_obs_t = torch.as_tensor(self._obs, device=device, dtype=torch.float32)
            final_value = policy.value(final_obs_t)
        finally:
            policy.train(was_training)

        advantages, returns = compute_gae(
            rewards=rewards_buf,
            values=values_buf,
            terminated=terminated_buf,
            truncated=truncated_buf,
            final_value=final_value,
            truncated_final_values=truncated_value_buf,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        flat_obs = obs_buf.reshape(T * E, *obs_shape)
        if self._action_is_discrete:
            flat_actions = actions_buf.reshape(T * E)
        else:
            flat_actions = actions_buf.reshape(T * E, -1)

        old_dist_params = _flatten_dist_params(
            kind="categorical" if self._action_is_discrete else "diag_gaussian",
            logits=logits_buf,
            mean=mean_buf,
            log_std=log_std_buf,
            t=T,
            e=E,
        )

        batch = RolloutBatch(
            obs=flat_obs,
            actions=flat_actions,
            logprobs=logp_buf.reshape(T * E),
            values=values_buf.reshape(T * E),
            rewards=rewards_buf.reshape(T * E),
            dones=torch.logical_or(terminated_buf, truncated_buf).reshape(T * E),
            advantages=advantages.reshape(T * E),
            returns=returns.reshape(T * E),
            old_dist_params=old_dist_params,
        )
        return batch, EpisodeStats(
            raw_returns=ep_raw_returns,
            wrapped_returns=ep_wrapped_returns,
            lengths=ep_lengths,
        )


def _flatten_dist_params(
    *,
    kind: str,
    logits: torch.Tensor | None,
    mean: torch.Tensor | None,
    log_std: torch.Tensor | None,
    t: int,
    e: int,
) -> PolicyDistParams:
    if kind == "categorical":
        assert logits is not None, "expected logits buffer for categorical policy"
        return PolicyDistParams(
            kind="categorical", logits=logits.reshape(t * e, -1).detach()
        )
    assert mean is not None and log_std is not None, (
        "expected (mean, log_std) buffers for diag-Gaussian policy"
    )
    return PolicyDistParams(
        kind="diag_gaussian",
        mean=mean.reshape(t * e, -1).detach(),
        log_std=log_std.reshape(t * e, -1).detach(),
    )


def _extract_raw_returns(info: Any, num_envs: int) -> list[float | None] | None:
    """
    Return a per-env list of raw episode returns (or None if not finished).

    Defensive across two gymnasium vector-env info shapes:

    - Newer (gymnasium >= 1.x): ``info["episode"]`` is a dict-of-arrays
      with ``"r"`` (returns) and ``"l"`` (lengths); ``info["_episode"]``
      is a boolean mask of which envs completed an episode this step.
    - Same-step autoreset: ``info["final_info"]`` may be a dict-of-arrays
      (Gymnasium 1.x) or a per-env list of dicts (Gymnasium 0.29.x).

    If neither shape is found, returns ``None`` and the caller falls
    back to the manually-accumulated wrapped return.
    """
    if not isinstance(info, dict):
        return None

    ep_info = info.get("episode")
    if isinstance(ep_info, dict) and "r" in ep_info:
        mask = info.get("_episode")
        rs = np.asarray(ep_info["r"]).reshape(-1)
        out: list[float | None] = [None] * num_envs
        for i in range(num_envs):
            if mask is None or bool(np.asarray(mask).reshape(-1)[i]):
                out[i] = float(rs[i])
        return out

    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        ep = final_info.get("episode")
        if isinstance(ep, dict) and "r" in ep:
            mask = final_info.get("_episode")
            if mask is None:
                mask = info.get("_final_info")
            rs = np.asarray(ep["r"]).reshape(-1)
            out_d: list[float | None] = [None] * num_envs
            for i in range(num_envs):
                if mask is None or bool(np.asarray(mask).reshape(-1)[i]):
                    out_d[i] = float(rs[i])
            return out_d

    if isinstance(final_info, (list, tuple, np.ndarray)):
        out_l: list[float | None] = [None] * num_envs
        for i, sub in enumerate(final_info):
            if isinstance(sub, dict) and "episode" in sub:
                ep = sub["episode"]
                out_l[i] = float(ep["r"])
        return out_l

    return None


def _extract_final_observations(
    info: Any, num_envs: int
) -> list[np.ndarray | None] | None:
    """Return final observations from same-step autoreset info, if present."""
    if not isinstance(info, dict):
        return None
    final_obs = info.get("final_obs")
    if final_obs is None:
        final_obs = info.get("final_observation")
    if final_obs is None:
        return None

    mask = info.get("_final_obs")
    if mask is None:
        mask = info.get("_final_observation")
    if mask is None:
        mask = info.get("_final_info")

    out: list[np.ndarray | None] = [None] * num_envs
    mask_arr = None if mask is None else np.asarray(mask).reshape(-1)
    if isinstance(final_obs, np.ndarray) and final_obs.dtype == object:
        for i in range(num_envs):
            if mask_arr is None or bool(mask_arr[i]):
                obs_i = final_obs[i]
                if obs_i is not None:
                    out[i] = np.asarray(obs_i, dtype=np.float32)
        return out

    obs_arr = np.asarray(final_obs)
    for i in range(num_envs):
        if mask_arr is None or bool(mask_arr[i]):
            out[i] = np.asarray(obs_arr[i], dtype=np.float32)
    return out
