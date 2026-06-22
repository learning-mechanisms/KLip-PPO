"""
Deliberate sweeps over algorithm-specific knobs.

For each swept env, we run three knob grids that target the same underlying
question — what trust region size is "right" for this objective and env:

  - ``ppo_kl_fixed.beta``        ∈ ``BETA_GRID``        (penalty scale),
  - ``ppo_kl_adaptive.kl_target`` ∈ ``KL_TARGET_GRID``  (target divergence),
  - ``ppo_clip.clip_epsilon``    ∈ ``CLIP_EPSILON_GRID`` (trust-region radius).

Sweeping ``clip_epsilon`` on the clip baseline keeps the comparison fair: if
β or kl_target are tuned per-env, ε should be too — otherwise the headline
table can silently advantage the KL variants just because PPO-Clip was left
at an uninformed default.

Envs covered:

  - **CartPole-v1** (smoke env, fast iteration; ``cc-sweeps``),
  - **Hopper-v4** (MuJoCo, largest expected ``I_kill`` activity; ``mujoco-sweeps``),
  - **HalfCheetah-v4** (MuJoCo, smooth dynamics, low seed variance — the
    representative locomotion env so the sweep response curves are diagnostic
    rather than dominated by Hopper's well-known instability; ``mujoco-sweeps``).

The intent is *not* to fit a final headline number — it is to pick a defensible
knob value per env from a small grid before the headline comparison runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from klip_ppo.configs.env import EnvConfig
from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.network import MLPConfig
from klip_ppo.configs.rollout import RolloutConfig
from klip_ppo.configs.trainer import TrainerConfig
from klip_ppo.experiments import box2d, classic_control, mujoco
from klip_ppo.experiments.common import (
    make_algorithm,
    make_experiment,
    shared_algo_knobs,
)

# Group names that materialise under configs/snapshots/presets/<group>/.
GROUP_CC = "cc-sweeps"
GROUP_MUJOCO = "mujoco-sweeps"

BETA_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
KL_TARGET_GRID: tuple[float, ...] = (0.003, 0.01, 0.02, 0.03, 0.1)
CLIP_EPSILON_GRID: tuple[float, ...] = (0.1, 0.2, 0.3)


def _fmt(value: float) -> str:
    """Slug-safe float formatting that preserves the value: 0.003 -> '0p003',
    0.1 -> '0p1', 1.0 -> '1', 10.0 -> '10'.

    Uses 4 decimal digits and trims trailing zeros / trailing dot, so every entry
    in the sweep grids slugs uniquely.
    """
    s = f"{value:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if s else "0"


@dataclass(frozen=True)
class _SweepEnv:
    """Per-env scaffolding for a sweep: builds env/network/rollout/trainer."""

    slug: str
    build_env: Callable[[], EnvConfig]
    build_network: Callable[[], MLPConfig]
    build_rollout: Callable[[], RolloutConfig]
    build_trainer: Callable[[], TrainerConfig]


def _cartpole_env_spec() -> _SweepEnv:
    return _SweepEnv(
        slug="cartpole",
        build_env=classic_control._env,
        build_network=classic_control._network,
        build_rollout=classic_control._rollout,
        build_trainer=classic_control._trainer,
    )


def _mujoco_env_spec(slug: str) -> _SweepEnv:
    spec = next(s for s in mujoco.SPECS if s.slug == slug)

    def _env(s: mujoco.MujocoEnvSpec = spec) -> EnvConfig:
        return mujoco._env(s)

    def _network(s: mujoco.MujocoEnvSpec = spec) -> MLPConfig:
        return mujoco._network(s)

    def _trainer(s: mujoco.MujocoEnvSpec = spec) -> TrainerConfig:
        return mujoco._trainer(s)

    return _SweepEnv(
        slug=slug,
        build_env=_env,
        build_network=_network,
        build_rollout=mujoco._rollout,
        build_trainer=_trainer,
    )


_CC_SWEEP_ENVS: tuple[_SweepEnv, ...] = (_cartpole_env_spec(),)
_MUJOCO_SWEEP_ENVS: tuple[_SweepEnv, ...] = (
    _mujoco_env_spec("hopper"),
    _mujoco_env_spec("halfcheetah"),
)


def _build_preset(env: _SweepEnv, name_suffix: str, algorithm) -> ExperimentConfig:
    """Construct one swept preset on ``env`` with a custom algorithm config."""
    return make_experiment(
        name=f"{env.slug}__{name_suffix}",
        env=env.build_env(),
        algorithm=algorithm,
        network=env.build_network(),
        rollout=env.build_rollout(),
        trainer=env.build_trainer(),
        tags=("ppo", "sweep", env.slug, name_suffix),
    )


def _beta_presets(
    envs: tuple[_SweepEnv, ...],
) -> dict[str, Callable[[], ExperimentConfig]]:
    out: dict[str, Callable[[], ExperimentConfig]] = {}
    for env in envs:
        for beta in BETA_GRID:
            suffix = f"ppo_kl_fixed__beta_{_fmt(beta)}"
            name = f"{env.slug}__{suffix}"

            def factory(
                b: float = beta, e: _SweepEnv = env, s: str = suffix
            ) -> ExperimentConfig:
                algo = make_algorithm("ppo_kl_fixed", shared_algo_knobs(), beta_fixed=b)
                return _build_preset(e, s, algo)

            out[name] = factory
    return out


def _kl_target_presets(
    envs: tuple[_SweepEnv, ...],
) -> dict[str, Callable[[], ExperimentConfig]]:
    out: dict[str, Callable[[], ExperimentConfig]] = {}
    for env in envs:
        for target in KL_TARGET_GRID:
            suffix = f"ppo_kl_adaptive__kl_target_{_fmt(target)}"
            name = f"{env.slug}__{suffix}"

            def factory(
                t: float = target, e: _SweepEnv = env, s: str = suffix
            ) -> ExperimentConfig:
                algo = make_algorithm(
                    "ppo_kl_adaptive", shared_algo_knobs(), kl_target=t
                )
                return _build_preset(e, s, algo)

            out[name] = factory
    return out


def _clip_epsilon_presets(
    envs: tuple[_SweepEnv, ...],
) -> dict[str, Callable[[], ExperimentConfig]]:
    out: dict[str, Callable[[], ExperimentConfig]] = {}
    for env in envs:
        for eps in CLIP_EPSILON_GRID:
            suffix = f"ppo_clip__clip_eps_{_fmt(eps)}"
            name = f"{env.slug}__{suffix}"

            def factory(
                ep: float = eps, e: _SweepEnv = env, s: str = suffix
            ) -> ExperimentConfig:
                algo = make_algorithm("ppo_clip", shared_algo_knobs(), clip_epsilon=ep)
                return _build_preset(e, s, algo)

            out[name] = factory
    return out


def cc_presets() -> dict[str, Callable[[], ExperimentConfig]]:
    """All CartPole sweep presets (β × kl_target × clip_ε)."""
    return {
        **_beta_presets(_CC_SWEEP_ENVS),
        **_kl_target_presets(_CC_SWEEP_ENVS),
        **_clip_epsilon_presets(_CC_SWEEP_ENVS),
    }


def mujoco_presets() -> dict[str, Callable[[], ExperimentConfig]]:
    """All MuJoCo sweep presets across {Hopper, HalfCheetah} × three knob grids."""
    return {
        **_beta_presets(_MUJOCO_SWEEP_ENVS),
        **_kl_target_presets(_MUJOCO_SWEEP_ENVS),
        **_clip_epsilon_presets(_MUJOCO_SWEEP_ENVS),
    }


# Silence the unused-import linter — box2d is reserved for future Box2D sweeps.
_ = box2d
