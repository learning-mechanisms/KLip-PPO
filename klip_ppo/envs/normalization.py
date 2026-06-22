"""Serializable observation and reward normalisation for rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RunningMeanStd:
    """Numerically stable running mean/variance estimator."""

    shape: tuple[int, ...] = ()
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[0] == 0:
            return
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        batch_count = float(arr.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = float(total_count)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "shape": self.shape,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
        self.shape = tuple(int(x) for x in state.get("shape", self.mean.shape))
        self.epsilon = float(state.get("epsilon", self.epsilon))


class EnvNormalizer:
    """Apply PPO-style obs/reward normalisation outside Gym wrappers."""

    def __init__(
        self,
        *,
        obs_shape: tuple[int, ...],
        num_envs: int,
        normalize_obs: bool,
        normalize_reward: bool,
        clip_obs: float,
        clip_reward: float,
        reward_scale: float,
        gamma: float,
        epsilon: float = 1e-8,
    ) -> None:
        self.obs_shape = obs_shape
        self.num_envs = int(num_envs)
        self.normalize_obs_enabled = bool(normalize_obs)
        self.normalize_reward_enabled = bool(normalize_reward)
        self.clip_obs = float(clip_obs)
        self.clip_reward = float(clip_reward)
        self.reward_scale = float(reward_scale)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs, dtype=np.float64)

    def normalize_obs(self, obs: np.ndarray, *, update: bool) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if not self.normalize_obs_enabled:
            return arr
        batched = arr.reshape(-1, *self.obs_shape)
        if update:
            self.obs_rms.update(batched)
        normalised = (arr - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )
        return np.clip(normalised, -self.clip_obs, self.clip_obs).astype(np.float32)

    def normalize_reward(
        self,
        reward: np.ndarray,
        *,
        terminated: np.ndarray,
        truncated: np.ndarray,
        update: bool,
    ) -> np.ndarray:
        arr = np.asarray(reward, dtype=np.float32)
        done = np.logical_or(terminated, truncated)
        if self.normalize_reward_enabled:
            self.returns = self.returns * self.gamma + arr.astype(np.float64)
            if update:
                self.return_rms.update(self.returns.reshape(-1))
            normalised = arr.astype(np.float64) / np.sqrt(
                self.return_rms.var + self.epsilon
            )
            arr = np.clip(normalised, -self.clip_reward, self.clip_reward).astype(
                np.float32
            )
            self.returns[done] = 0.0
        if self.reward_scale != 1.0:
            arr = (arr * self.reward_scale).astype(np.float32)
        return arr.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "obs_shape": self.obs_shape,
            "num_envs": self.num_envs,
            "normalize_obs": self.normalize_obs_enabled,
            "normalize_reward": self.normalize_reward_enabled,
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
            "reward_scale": self.reward_scale,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "obs_rms": self.obs_rms.state_dict(),
            "return_rms": self.return_rms.state_dict(),
            "returns": self.returns,
        }

    def load_state_dict(
        self, state: dict[str, Any], *, num_envs: int | None = None
    ) -> None:
        self.obs_shape = tuple(int(x) for x in state["obs_shape"])
        self.num_envs = int(num_envs if num_envs is not None else state["num_envs"])
        self.normalize_obs_enabled = bool(state["normalize_obs"])
        self.normalize_reward_enabled = bool(state["normalize_reward"])
        self.clip_obs = float(state["clip_obs"])
        self.clip_reward = float(state["clip_reward"])
        self.reward_scale = float(state["reward_scale"])
        self.gamma = float(state["gamma"])
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.obs_rms = RunningMeanStd(shape=self.obs_shape)
        self.obs_rms.load_state_dict(state["obs_rms"])
        self.return_rms = RunningMeanStd(shape=())
        self.return_rms.load_state_dict(state["return_rms"])
        saved_returns = np.asarray(state.get("returns", np.zeros(self.num_envs)))
        self.returns = np.zeros(self.num_envs, dtype=np.float64)
        n = min(self.num_envs, int(saved_returns.shape[0]))
        if n:
            self.returns[:n] = saved_returns[:n]

    @classmethod
    def from_state(cls, state: dict[str, Any], *, num_envs: int) -> EnvNormalizer:
        normalizer = cls(
            obs_shape=tuple(int(x) for x in state["obs_shape"]),
            num_envs=num_envs,
            normalize_obs=bool(state["normalize_obs"]),
            normalize_reward=bool(state["normalize_reward"]),
            clip_obs=float(state["clip_obs"]),
            clip_reward=float(state["clip_reward"]),
            reward_scale=float(state["reward_scale"]),
            gamma=float(state["gamma"]),
            epsilon=float(state.get("epsilon", 1e-8)),
        )
        normalizer.load_state_dict(state, num_envs=num_envs)
        return normalizer
