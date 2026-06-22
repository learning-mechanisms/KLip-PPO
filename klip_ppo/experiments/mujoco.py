"""
MuJoCo locomotion presets (5 envs × 4 PPO variants).

Hyperparameters are the canonical default-scale PPO MuJoCo setup, matching
Schulman 2017, Engstrom 2020, and Andrychowicz 2021:

  - ``num_envs=1, n_steps=2048`` (so global batch / iter = 2048),
  - ``minibatch_size=64, epochs=10``,
  - ``gamma=0.99, gae_lambda=0.95``,
  - ``vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5``,
  - ``clip_value_loss=true, value_clip_epsilon=0.2``,
  - obs + reward normalisation on,
  - Adam ``lr=3e-4, eps=1e-5``,
  - rollout-scope advantage normalisation (CleanRL convention),
  - tanh MLP [64, 64] across all five envs (including Humanoid),
  - 1M environment steps per run.

The per-env base is shared across the four variants, by construction switching
algorithm cannot change env / network / rollout / trainer knobs.

Evaluation protocol:
  Periodic deterministic evaluation is enabled at ``eval_every_steps=50_000``
  with ``eval_episodes=10`` so post-update ``eval/return/*`` series are available
  for the benchmark table. The paper's headline figures still use training-
  rollout returns (paper §4.2 "their return curves and clip-fraction traces
  overlay"); eval is a supplement that gives a literature-comparable final-
  performance number free of reward normalisation and rolling-window smoothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from klip_ppo.configs.env import EnvConfig
from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.network import MLPConfig
from klip_ppo.configs.rollout import RolloutConfig
from klip_ppo.configs.trainer import TrainerConfig
from klip_ppo.experiments.common import (
    ALGO_KINDS,
    AlgoKind,
    algo_tag,
    make_algorithm,
    make_experiment,
    shared_algo_knobs,
)

GROUP = "mujoco-baselines"


@dataclass(frozen=True)
class MujocoEnvSpec:
    """Per-env knobs that differ across the locomotion suite."""

    env_id: str
    slug: str
    total_steps: int
    hidden_sizes: tuple[int, ...] = (64, 64)


SPECS: tuple[MujocoEnvSpec, ...] = (
    MujocoEnvSpec(env_id="Hopper-v4", slug="hopper", total_steps=1_000_000),
    MujocoEnvSpec(env_id="HalfCheetah-v4", slug="halfcheetah", total_steps=1_000_000),
    MujocoEnvSpec(env_id="Walker2d-v4", slug="walker2d", total_steps=1_000_000),
    MujocoEnvSpec(env_id="Humanoid-v4", slug="humanoid", total_steps=1_000_000),
    MujocoEnvSpec(env_id="Ant-v4", slug="ant", total_steps=1_000_000),
)


def _env(spec: MujocoEnvSpec) -> EnvConfig:
    return EnvConfig(
        id=spec.env_id,
        normalize_obs=True,
        normalize_reward=True,
    )


def _network(spec: MujocoEnvSpec) -> MLPConfig:
    return MLPConfig(
        hidden_sizes=spec.hidden_sizes,
        activation="tanh",
        ortho_init=True,
        log_std_init=0.0,
    )


def _rollout() -> RolloutConfig:
    return RolloutConfig(num_envs=1, n_steps=2048, async_envs=False)


def _trainer(spec: MujocoEnvSpec) -> TrainerConfig:
    return TrainerConfig(
        total_steps=spec.total_steps,
        log_every_iters=1,
        eval_every_steps=50_000,
        eval_episodes=10,
        eval_deterministic=True,
    )


def mujoco(spec: MujocoEnvSpec, kind: AlgoKind) -> ExperimentConfig:
    """Build one MuJoCo preset for one (env, algorithm) pair."""
    shared = shared_algo_knobs()
    algorithm = make_algorithm(kind, shared)
    return make_experiment(
        name=f"{spec.slug}__{kind}",
        env=_env(spec),
        algorithm=algorithm,
        network=_network(spec),
        rollout=_rollout(),
        trainer=_trainer(spec),
        tags=("ppo", algo_tag(kind), "mujoco", spec.slug),
    )


def presets() -> dict[str, Callable[[], ExperimentConfig]]:
    out: dict[str, Callable[[], ExperimentConfig]] = {}
    for spec in SPECS:
        for kind in ALGO_KINDS:
            name = f"{spec.slug}__{kind}"

            def factory(
                s: MujocoEnvSpec = spec, k: AlgoKind = kind
            ) -> ExperimentConfig:
                return mujoco(s, k)

            out[name] = factory
    return out
