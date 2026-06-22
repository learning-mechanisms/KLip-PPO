"""Soft-clipping presets for workshop screening and confirmation runs.

Reader note: at a shared ``softness``, the three methods do not apply the same
amount of softening. ``sigmoid`` has a nonzero inside-band tail (at
``softness = 0.05`` it brakes ~12% of the unclipped gradient at ratio 1.1,
inside I_in), while ``linear_ramp`` and ``soft_min`` do not. The presets
are matched on the public ``softness`` knob, not on the inside-band gate
mean. See ``PPOSoftClipConfig`` docstring and
``soft_clip/gate/mean/I_in`` / ``soft_clip/gate/mean/I_pass`` in the logs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from klip_ppo.configs.algorithm.ppo_soft_clip import SoftClipMethod
from klip_ppo.configs.env import EnvConfig
from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.network import MLPConfig
from klip_ppo.configs.rollout import RolloutConfig
from klip_ppo.configs.trainer import TrainerConfig
from klip_ppo.experiments import classic_control, mujoco
from klip_ppo.experiments.common import (
    make_algorithm,
    make_experiment,
    shared_algo_knobs,
)

GROUP_CC = "cc-soft-clipping"
GROUP_MUJOCO = "mujoco-soft-clipping"
SOFTNESS_GRID: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)
SOFTNESS_DEFAULT = 0.05
SOFT_METHODS: tuple[SoftClipMethod, ...] = ("linear_ramp", "sigmoid", "soft_min")


def _fmt(value: float) -> str:
    s = f"{value:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if s else "0"


@dataclass(frozen=True)
class _SoftClipEnv:
    """Per-env scaffolding for soft-clipping presets."""

    slug: str
    build_env: Callable[[], EnvConfig]
    build_network: Callable[[], MLPConfig]
    build_rollout: Callable[[], RolloutConfig]
    build_trainer: Callable[[], TrainerConfig]


def _cartpole_env_spec() -> _SoftClipEnv:
    return _SoftClipEnv(
        slug="cartpole",
        build_env=classic_control._env,
        build_network=classic_control._network,
        build_rollout=classic_control._rollout,
        build_trainer=classic_control._trainer,
    )


def _mujoco_env_spec(slug: str) -> _SoftClipEnv:
    spec = next(s for s in mujoco.SPECS if s.slug == slug)

    def _env(s: mujoco.MujocoEnvSpec = spec) -> EnvConfig:
        return mujoco._env(s)

    def _network(s: mujoco.MujocoEnvSpec = spec) -> MLPConfig:
        return mujoco._network(s)

    def _rollout() -> RolloutConfig:
        return mujoco._rollout()

    def _trainer(s: mujoco.MujocoEnvSpec = spec) -> TrainerConfig:
        return mujoco._trainer(s)

    return _SoftClipEnv(
        slug=slug,
        build_env=_env,
        build_network=_network,
        build_rollout=_rollout,
        build_trainer=_trainer,
    )


_CC_ENVS: tuple[_SoftClipEnv, ...] = (_cartpole_env_spec(),)
_MUJOCO_ENVS: tuple[_SoftClipEnv, ...] = (
    _mujoco_env_spec("hopper"),
    _mujoco_env_spec("halfcheetah"),
)


def _soft_clip_preset(
    env: _SoftClipEnv, method: SoftClipMethod, *, softness: float
) -> ExperimentConfig:
    """Build one soft-clipping preset with the env's baseline knobs."""
    algorithm = make_algorithm(
        "ppo_soft_clip",
        shared_algo_knobs(),
        soft_method=method,
        softness=softness,
    )
    softness_slug = _fmt(softness)
    return make_experiment(
        name=f"{env.slug}__ppo_soft_clip__{method}_s{softness_slug}",
        env=env.build_env(),
        algorithm=algorithm,
        network=env.build_network(),
        rollout=env.build_rollout(),
        trainer=env.build_trainer(),
        tags=("ppo", "soft_clip", method, env.slug),
    )


def _presets_for(
    envs: tuple[_SoftClipEnv, ...],
) -> dict[str, Callable[[], ExperimentConfig]]:
    out: dict[str, Callable[[], ExperimentConfig]] = {}
    for env in envs:
        for method in SOFT_METHODS:
            for softness in SOFTNESS_GRID:
                name = f"{env.slug}__ppo_soft_clip__{method}_s{_fmt(softness)}"

                def factory(
                    e: _SoftClipEnv = env,
                    m: SoftClipMethod = method,
                    s: float = softness,
                ) -> ExperimentConfig:
                    return _soft_clip_preset(e, m, softness=s)

                out[name] = factory
    return out


def cartpole_soft_clip(
    method: SoftClipMethod, *, softness: float = SOFTNESS_DEFAULT
) -> ExperimentConfig:
    """Build one CartPole soft-clipping preset with CartPole baseline knobs."""
    return _soft_clip_preset(_cartpole_env_spec(), method, softness=softness)


def cc_presets() -> dict[str, Callable[[], ExperimentConfig]]:
    """CartPole soft-clipping presets across methods and softness values."""
    return _presets_for(_CC_ENVS)


def mujoco_presets() -> dict[str, Callable[[], ExperimentConfig]]:
    """MuJoCo soft-clipping presets across Hopper and HalfCheetah."""
    return _presets_for(_MUJOCO_ENVS)
