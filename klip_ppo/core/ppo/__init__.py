"""PPO trainer plus per-variant loss strategies."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ClipStrategy",
    "EpochAggregate",
    "KLAdaptiveStrategy",
    "KLFixedStrategy",
    "KLPerSampleStrategy",
    "PPOTrainer",
    "RunResult",
    "Strategy",
    "StrategyOutputs",
    "build_strategy",
]


def __getattr__(name: str) -> Any:
    """Load package-level PPO exports without eager trainer imports."""
    if name in {"PPOTrainer", "RunResult"}:
        from klip_ppo.core.ppo.trainer import PPOTrainer, RunResult

        return {"PPOTrainer": PPOTrainer, "RunResult": RunResult}[name]
    if name in {
        "ClipStrategy",
        "EpochAggregate",
        "KLAdaptiveStrategy",
        "KLFixedStrategy",
        "KLPerSampleStrategy",
        "Strategy",
        "StrategyOutputs",
        "build_strategy",
    }:
        from klip_ppo.core.ppo.strategies import (
            ClipStrategy,
            EpochAggregate,
            KLAdaptiveStrategy,
            KLFixedStrategy,
            KLPerSampleStrategy,
            Strategy,
            StrategyOutputs,
            build_strategy,
        )

        return {
            "ClipStrategy": ClipStrategy,
            "EpochAggregate": EpochAggregate,
            "KLAdaptiveStrategy": KLAdaptiveStrategy,
            "KLFixedStrategy": KLFixedStrategy,
            "KLPerSampleStrategy": KLPerSampleStrategy,
            "Strategy": Strategy,
            "StrategyOutputs": StrategyOutputs,
            "build_strategy": build_strategy,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
