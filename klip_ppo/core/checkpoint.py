"""
Checkpoint manager.

Writes a single ``.pt`` blob per checkpoint containing model, optimiser, scheduler,
strategy, collector, and RNG state. ``final.pt`` is always written at end-of-run when
``save_final=True``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from klip_ppo.core.ppo.strategies.base import Strategy


class CheckpointManager:
    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        name: str,
        iteration: int,
        env_step: int,
        model: nn.Module,
        optim: Optimizer,
        scheduler: LRScheduler | None,
        strategy: Strategy,
        collector_state: dict[str, Any] | None = None,
    ) -> Path:
        path = self.dir / f"{name}.pt"
        torch.save(
            {
                "iteration": iteration,
                "env_step": env_step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "strategy": strategy.state_dict(),
                "collector": collector_state,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            },
            path,
        )
        return path

    def maybe_save_periodic(
        self,
        *,
        every_steps: int | None,
        env_step: int,
        prev_env_step: int,
        iteration: int,
        model: nn.Module,
        optim: Optimizer,
        scheduler: LRScheduler | None,
        strategy: Strategy,
        collector_state: dict[str, Any] | None = None,
    ) -> Path | None:
        if every_steps is None:
            return None
        if (env_step // every_steps) > (prev_env_step // every_steps):
            return self.save(
                name=f"policy_step_{env_step}",
                iteration=iteration,
                env_step=env_step,
                model=model,
                optim=optim,
                scheduler=scheduler,
                strategy=strategy,
                collector_state=collector_state,
            )
        return None

    def load(self, path: Path) -> dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=False)
