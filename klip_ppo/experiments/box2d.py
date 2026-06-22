"""
Box2D presets (LunarLander-v3 × 4 PPO variants).

LunarLander-v2 was deprecated in Gymnasium 1.x; we use the current ``v3``.
Hyperparameters mostly follow the SB3 / RL Zoo LunarLander baseline:

  - ``gamma=0.999, gae_lambda=0.98`` (longer credit assignment than MuJoCo
    defaults; matches the SB3 zoo and CleanRL LunarLander script),
  - ``ent_coef=0.01`` (small entropy bonus stabilises discrete-action PPO),
  - 16 sync envs × 1024 steps per iter = 16384 step batch,
  - 1M total env steps,
  - obs / reward normalisation off (discrete-action env, raw rewards are
    bounded and well-scaled).

We intentionally retain the shared ``epochs=10`` PPO setting for parity with the
other variants in this repository, while RL Zoo's tuned LunarLander-v3 entry uses
``n_epochs=4``.
"""

from __future__ import annotations

from collections.abc import Callable

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

GROUP = "box2d-baselines"
_ENV_ID = "LunarLander-v3"
_ENV_SLUG = "lunarlander"


def _env() -> EnvConfig:
    return EnvConfig(
        id=_ENV_ID,
        normalize_obs=False,
        normalize_reward=False,
    )


def _network() -> MLPConfig:
    return MLPConfig(hidden_sizes=(64, 64), activation="tanh", ortho_init=True)


def _rollout() -> RolloutConfig:
    # async_envs would be a nice speedup, but tests on Box2D under multiprocess
    # have been flaky on macOS; keep sync for portability.
    return RolloutConfig(num_envs=16, n_steps=1024, async_envs=False)


def _trainer() -> TrainerConfig:
    return TrainerConfig(total_steps=1_000_000, log_every_iters=1)


def lunarlander(kind: AlgoKind) -> ExperimentConfig:
    shared = shared_algo_knobs(
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
    )
    algorithm = make_algorithm(kind, shared)
    return make_experiment(
        name=f"{_ENV_SLUG}__{kind}",
        env=_env(),
        algorithm=algorithm,
        network=_network(),
        rollout=_rollout(),
        trainer=_trainer(),
        tags=("ppo", algo_tag(kind), "box2d", _ENV_SLUG),
    )


def presets() -> dict[str, Callable[[], ExperimentConfig]]:
    out: dict[str, Callable[[], ExperimentConfig]] = {}
    for kind in ALGO_KINDS:

        def factory(k: AlgoKind = kind) -> ExperimentConfig:
            return lunarlander(k)

        out[f"{_ENV_SLUG}__{kind}"] = factory
    return out
