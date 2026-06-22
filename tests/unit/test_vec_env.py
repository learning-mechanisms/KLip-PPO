"""Collector semantics around vector env autoreset and time limits."""

from __future__ import annotations

import torch
from klip_ppo.configs.env import EnvConfig
from klip_ppo.configs.network import MLPConfig
from klip_ppo.configs.rollout import RolloutConfig
from klip_ppo.core.networks import ActorCritic
from klip_ppo.envs.gym_env import probe_spaces
from klip_ppo.envs.vec_env import VectorCollector


def test_time_limit_steps_do_not_insert_dummy_reset_transition():
    env_cfg = EnvConfig(id="CartPole-v1", max_episode_steps=2)
    rollout_cfg = RolloutConfig(num_envs=1, n_steps=4, async_envs=False)
    obs_space, act_space = probe_spaces(env_cfg)
    model = ActorCritic(obs_space, act_space, MLPConfig())  # type: ignore[arg-type]
    collector = VectorCollector(
        env_cfg,
        rollout_cfg,
        gamma=0.99,
        gae_lambda=0.95,
        device=torch.device("cpu"),
        seed=0,
    )

    try:
        rollout, ep_stats = collector.collect(model)
    finally:
        collector.close()

    torch.testing.assert_close(rollout.rewards, torch.ones_like(rollout.rewards))
    assert int(rollout.dones.sum().item()) == 2
    assert ep_stats.raw_returns == [2.0, 2.0]
    assert ep_stats.wrapped_returns == [2.0, 2.0]
