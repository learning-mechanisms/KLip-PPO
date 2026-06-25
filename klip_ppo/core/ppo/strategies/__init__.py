"""PPO loss strategies — the only per-variant component in the trainer."""

from klip_ppo.configs.algorithm import AnyAlgorithmConfig
from klip_ppo.core.ppo.strategies.base import (
    EpochAggregate,
    Strategy,
    StrategyOutputs,
)
from klip_ppo.core.ppo.strategies.clip import ClipStrategy
from klip_ppo.core.ppo.strategies.kl_adaptive import KLAdaptiveStrategy
from klip_ppo.core.ppo.strategies.kl_fixed import KLFixedStrategy
from klip_ppo.core.ppo.strategies.kl_per_sample import KLPerSampleStrategy
from klip_ppo.core.ppo.strategies.soft_clip import SoftClipStrategy


def build_strategy(algorithm: AnyAlgorithmConfig) -> Strategy:
    kind = algorithm.kind
    if kind == "ppo_clip":
        return ClipStrategy(algorithm)  # type: ignore[arg-type]
    if kind == "ppo_kl_fixed":
        return KLFixedStrategy(algorithm)  # type: ignore[arg-type]
    if kind == "ppo_kl_adaptive":
        return KLAdaptiveStrategy(algorithm)  # type: ignore[arg-type]
    if kind == "ppo_kl_per_sample":
        return KLPerSampleStrategy(algorithm)  # type: ignore[arg-type]
    if kind == "ppo_soft_clip":
        return SoftClipStrategy(algorithm)  # type: ignore[arg-type]
    raise ValueError(f"unknown algorithm kind: {kind!r}")


__all__ = [
    "ClipStrategy",
    "EpochAggregate",
    "KLAdaptiveStrategy",
    "KLFixedStrategy",
    "KLPerSampleStrategy",
    "SoftClipStrategy",
    "Strategy",
    "StrategyOutputs",
    "build_strategy",
]
