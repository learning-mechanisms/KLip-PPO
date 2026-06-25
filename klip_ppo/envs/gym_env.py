"""Gymnasium env construction with the wrappers we standardise on."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym

from klip_ppo.configs.env import EnvConfig


def make_env(cfg: EnvConfig, *, seed: int, env_idx: int) -> Callable[[], gym.Env]:
    """
    Return a zero-arg thunk that builds a single env instance.

    The same thunks feed both ``SyncVectorEnv`` and ``AsyncVectorEnv``. Seeding is done
    inside the thunk so async subprocesses get distinct streams.
    """

    def _thunk() -> gym.Env:
        env: gym.Env = gym.make(cfg.id)
        if cfg.max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        # Keep Gym rewards/observations raw. The rollout collector owns
        # normalisation so its running statistics can be checkpointed and reused.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, gym.spaces.Box):
            # Matches the CleanRL / SB3 convention but is not mentioned in Schulman
            # 2017. The rollout buffer stores the raw sampled action and its raw
            # log-prob while the env steps the clipped action, so PPO importance
            # ratios are biased at samples where the action saturates the box.
            # The bias is identical across all four PPO variants in this repo, so
            # it does not break the per-sample equivalence claim.
            env = gym.wrappers.ClipAction(env)
        env.action_space.seed(seed + env_idx)
        env.observation_space.seed(seed + env_idx)
        return env

    return _thunk


def probe_spaces(cfg: EnvConfig) -> tuple[gym.spaces.Space, gym.spaces.Space]:
    """Build one env briefly to read its observation / action spaces."""
    env = make_env(cfg, seed=0, env_idx=0)()
    try:
        return env.observation_space, env.action_space
    finally:
        env.close()
