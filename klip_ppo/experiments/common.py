"""
Shared building blocks for the experiment registry.

Every preset across every env / algorithm runs through this module to construct its
``ExperimentConfig``. That is the parity guarantee: variants on the same env share
*exactly* the same env / network / rollout / trainer settings and the same shared
algorithm knobs (epochs, gamma, gae_lambda, minibatch_size, optimiser, value-clip,
etc.). Variants only differ in their objective-specific fields (clip_epsilon, β,
β_init/kl_target, or soft-clipping method/softness).

If a knob should be common across all variants on an env, set it here or on the per-env
base. If it's variant-specific, set it on the per-variant factory.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from klip_ppo.configs.algorithm.base import AdvantageNormalization
from klip_ppo.configs.algorithm.ppo_clip import PPOClipConfig
from klip_ppo.configs.algorithm.ppo_kl_adaptive import PPOKLAdaptiveConfig
from klip_ppo.configs.algorithm.ppo_kl_fixed import PPOKLFixedConfig
from klip_ppo.configs.algorithm.ppo_kl_per_sample import PPOKLPerSampleConfig
from klip_ppo.configs.algorithm.ppo_soft_clip import PPOSoftClipConfig, SoftClipMethod
from klip_ppo.configs.env import EnvConfig
from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.logging_cfg import LoggingConfig
from klip_ppo.configs.network import MLPConfig, OptimiserConfig
from klip_ppo.configs.rollout import RolloutConfig
from klip_ppo.configs.trainer import TrainerConfig

AlgoKind = Literal[
    "ppo_clip",
    "ppo_kl_fixed",
    "ppo_kl_adaptive",
    "ppo_kl_per_sample",
    "ppo_soft_clip",
]
KLPenalty = Literal["full", "sample", "k3"]

# Baseline suites intentionally remain the four established hard/KL variants.
# Soft-clipping presets live in a dedicated experiment group so adding the new
# algorithm does not silently broaden every headline baseline.
ALGO_KINDS: tuple[AlgoKind, ...] = (
    "ppo_clip",
    "ppo_kl_fixed",
    "ppo_kl_adaptive",
    "ppo_kl_per_sample",
)


def shared_algo_knobs(
    *,
    epochs: int = 10,
    minibatch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    advantage_normalization: AdvantageNormalization = "rollout",
    clip_value_loss: bool = True,
    value_clip_epsilon: float = 0.2,
    lr: float = 3e-4,
    adam_eps: float = 1e-5,
    anneal_lr: bool = True,
) -> dict[str, Any]:
    """
    Hyperparameters shared by every PPO variant on a given env.

    Returned as a dict so the four variant builders can splat it. Centralising the
    defaults here is the only way to keep parity across variants honest: a knob set
    once flows into all four configs by construction.

    ``anneal_lr`` defaults to ``True`` here to match the CleanRL PPO baseline
    (linear decay from ``lr`` to ``0`` over the training run); the field default on
    ``OptimiserConfig`` itself is ``False`` so isolated raw-config tests keep the
    constant-LR baseline.
    """
    return {
        "epochs": epochs,
        "minibatch_size": minibatch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "vf_coef": vf_coef,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "advantage_normalization": advantage_normalization,
        "clip_value_loss": clip_value_loss,
        "value_clip_epsilon": value_clip_epsilon,
        "optimiser": OptimiserConfig(lr=lr, eps=adam_eps, anneal_lr=anneal_lr),
    }


def make_algorithm(
    kind: AlgoKind,
    shared: dict[str, Any],
    *,
    clip_epsilon: float = 0.2,
    beta_fixed: float = 1.0,
    beta_init: float = 1.0,
    kl_target: float = 0.02,
    kl_penalty: KLPenalty = "full",
    soft_method: SoftClipMethod = "linear_ramp",
    softness: float = 0.05,
) -> (
    PPOClipConfig
    | PPOKLFixedConfig
    | PPOKLAdaptiveConfig
    | PPOKLPerSampleConfig
    | PPOSoftClipConfig
):
    """
    Build one algorithm config from ``shared`` plus variant-specific knobs.

    Defaults for variant-specific knobs (``beta_fixed``, ``kl_target``) are
    chosen so that the four objectives target roughly the same trust region:

      - ``clip_epsilon = 0.2`` (PPO paper default).
      - ``beta_fixed = 1.0`` (PPO paper §2 reference value).
      - ``kl_target = 0.02`` ≈ ``clip_epsilon**2 / 2``, which is the
        second-order-Taylor KL associated with a one-sided ratio bound of ε.
        The PPO paper's adaptive-KL sweep includes ``0.01`` as the strongest
        reported ``dtarg`` row; we use 0.02 here so PPO-KL-adaptive does not run
        a *tighter* trust region than PPO-Clip and bias the comparison.

    The β / kl_target sweeps in ``sweeps.py`` override these.
    """
    base = deepcopy(shared)
    if kind == "ppo_clip":
        return PPOClipConfig(**base, clip_epsilon=clip_epsilon)
    if kind == "ppo_kl_fixed":
        return PPOKLFixedConfig(
            **base,
            beta=beta_fixed,
            kl_penalty=kl_penalty,
            clip_epsilon_for_diagnostics=clip_epsilon,
        )
    if kind == "ppo_kl_adaptive":
        return PPOKLAdaptiveConfig(
            **base,
            beta_init=beta_init,
            kl_target=kl_target,
            kl_penalty=kl_penalty,
            clip_epsilon_for_diagnostics=clip_epsilon,
        )
    if kind == "ppo_kl_per_sample":
        return PPOKLPerSampleConfig(**base, clip_epsilon=clip_epsilon)
    if kind == "ppo_soft_clip":
        return PPOSoftClipConfig(
            **base,
            clip_epsilon=clip_epsilon,
            method=soft_method,
            softness=softness,
        )
    raise ValueError(f"unknown algorithm kind: {kind!r}")


def algo_tag(kind: AlgoKind) -> str:
    """Short tag used in WandB / log identifiers, matching existing YAML conventions."""
    return {
        "ppo_clip": "clip",
        "ppo_kl_fixed": "kl_fixed",
        "ppo_kl_adaptive": "kl_adaptive",
        "ppo_kl_per_sample": "kl_per_sample",
        "ppo_soft_clip": "soft_clip",
    }[kind]


def make_experiment(
    *,
    name: str,
    env: EnvConfig,
    algorithm: PPOClipConfig
    | PPOKLFixedConfig
    | PPOKLAdaptiveConfig
    | PPOKLPerSampleConfig
    | PPOSoftClipConfig,
    network: MLPConfig,
    rollout: RolloutConfig,
    trainer: TrainerConfig,
    tags: tuple[str, ...],
    seed: int = 0,
    logging: LoggingConfig | None = None,
) -> ExperimentConfig:
    """
    Final constructor — runs Pydantic validation.

    Any drift fails here.
    """
    return ExperimentConfig(
        name=name,
        seed=seed,
        algorithm=algorithm,
        env=env,
        network=network,
        rollout=rollout,
        trainer=trainer,
        logging=logging or LoggingConfig(),
        tags=tags,
    )
