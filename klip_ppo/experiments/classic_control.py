"""
Classic-control benchmark presets (CartPole-v1, four PPO variants).

All four variants are constructed from a single shared base (env / network / rollout /
trainer) so they cannot drift on shared knobs; only the algorithm-specific knobs differ.

Hyperparameters: SB3/CleanRL CartPole defaults — small net, short rollout, no env
normalisation, 100k env steps. CartPole-v1 is the smoke-test env.
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

GROUP = "cc-baselines"
_ENV_ID = "CartPole-v1"
_ENV_SLUG = "cartpole"


def _env() -> EnvConfig:
    return EnvConfig(
        id=_ENV_ID,
        normalize_obs=False,
        normalize_reward=False,
    )


def _network() -> MLPConfig:
    return MLPConfig(hidden_sizes=(64, 64), activation="tanh", ortho_init=True)


def _rollout() -> RolloutConfig:
    return RolloutConfig(num_envs=4, n_steps=128, async_envs=False)


def _trainer() -> TrainerConfig:
    return TrainerConfig(total_steps=100_000, log_every_iters=1)


def cartpole(kind: AlgoKind) -> ExperimentConfig:
    """Build one CartPole preset for the requested algorithm variant."""
    shared = shared_algo_knobs()
    algorithm = make_algorithm(kind, shared)
    return make_experiment(
        name=f"{_ENV_SLUG}__{kind}",
        env=_env(),
        algorithm=algorithm,
        network=_network(),
        rollout=_rollout(),
        trainer=_trainer(),
        tags=("ppo", algo_tag(kind), _ENV_SLUG),
    )


def presets() -> dict[str, Callable[[], ExperimentConfig]]:
    """
    Map preset name -> factory.

    Lazy so import-time cost stays low.
    """
    out: dict[str, Callable[[], ExperimentConfig]] = {}
    for kind in ALGO_KINDS:

        def factory(k: AlgoKind = kind) -> ExperimentConfig:
            return cartpole(k)

        out[f"{_ENV_SLUG}__{kind}"] = factory
    return out
