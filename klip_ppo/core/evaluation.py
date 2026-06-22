"""Policy evaluation helpers that keep benchmark returns raw."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.core.networks import ActorCritic
from klip_ppo.envs.gym_env import make_env
from klip_ppo.envs.normalization import EnvNormalizer


@dataclass(frozen=True)
class EvalStats:
    returns: tuple[float, ...]
    lengths: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.returns)

    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.returns else float("nan")

    def std_return(self) -> float:
        return float(np.std(self.returns)) if self.returns else float("nan")

    def iqm_return(self) -> float:
        if len(self.returns) < 4:
            return self.mean_return()
        arr = np.asarray(self.returns, dtype=np.float64)
        lo, hi = np.quantile(arr, [0.25, 0.75])
        mask = (arr >= lo) & (arr <= hi)
        return float(arr[mask].mean()) if mask.any() else float(arr.mean())

    def mean_length(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else float("nan")

    def as_log_row(self) -> dict[str, float | int]:
        return {
            "eval/return/mean": self.mean_return(),
            "eval/return/std": self.std_return(),
            "eval/return/iqm": self.iqm_return(),
            "eval/episode/len_mean": self.mean_length(),
            "eval/episode/count": self.n,
        }


def evaluate_policy(
    cfg: ExperimentConfig,
    model: ActorCritic,
    *,
    episodes: int,
    deterministic: bool,
    seed: int,
    device: torch.device,
    normalizer_state: dict[str, Any] | None = None,
) -> EvalStats:
    """
    Evaluate a policy and return raw Gym rewards.

    Observation normalisation reuses frozen training statistics when enabled. Reward
    normalisation is intentionally not applied: eval returns should be comparable to
    benchmark raw environment returns.
    """
    if cfg.env.normalize_obs and normalizer_state is None:
        raise ValueError(
            "cannot evaluate normalized-observation policy without checkpointed "
            "normalizer state"
        )
    normalizer = (
        EnvNormalizer.from_state(normalizer_state, num_envs=1)
        if normalizer_state is not None
        else None
    )
    env = make_env(cfg.env, seed=seed, env_idx=0)()
    returns: list[float] = []
    lengths: list[int] = []
    was_training = model.training
    model.eval()
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_return = 0.0
            ep_len = 0
            done = False
            while not done:
                obs_arr = np.asarray(obs, dtype=np.float32)
                if normalizer is not None:
                    obs_arr = normalizer.normalize_obs(
                        obs_arr[None, ...], update=False
                    )[0]
                obs_t = torch.as_tensor(obs_arr, device=device, dtype=torch.float32)
                action, _, _, _ = model.act(
                    obs_t.unsqueeze(0), deterministic=deterministic
                )
                env_action = _to_env_action(action.squeeze(0), env.action_space)
                obs, reward, terminated, truncated, _ = env.step(env_action)
                ep_return += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)
            returns.append(ep_return)
            lengths.append(ep_len)
    finally:
        model.train(was_training)
        env.close()
    return EvalStats(returns=tuple(returns), lengths=tuple(lengths))


def _to_env_action(
    action: torch.Tensor, action_space: gym.spaces.Space
) -> int | np.ndarray:
    action_array = action.detach().cpu().numpy()
    if isinstance(action_space, gym.spaces.Discrete):
        return int(action_array)
    return np.asarray(action_array, dtype=np.float32)
