"""Trainer-loop configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from klip_ppo.configs.base import BaseConfig

DiagnosticMode = Literal["standard", "full"]


class TrainerConfig(BaseConfig):
    """Outer-loop schedule for the PPO trainer."""

    total_steps: Annotated[int, Field(gt=0)]
    """Total environment transitions to collect before stopping."""

    eval_every_steps: Annotated[int, Field(gt=0)] | None = None
    """Environment-step interval between evaluation runs."""

    eval_episodes: Annotated[int, Field(gt=0)] = 10
    """Number of episodes per evaluation run."""

    eval_deterministic: bool = True
    """Whether evaluation uses mean actions instead of sampling."""

    checkpoint_every_steps: Annotated[int, Field(gt=0)] | None = None
    """Environment-step interval between checkpoint writes."""

    log_every_iters: Annotated[int, Field(gt=0)] = 1
    """Training-iteration interval between metric writes."""

    save_final_checkpoint: bool = True
    """Whether to write a checkpoint after training completes."""

    diagnostic_mode: DiagnosticMode = "standard"
    """Controls whether expensive per-inner-epoch diagnostics are emitted."""

    @model_validator(mode="after")
    def _supported_fields(self) -> TrainerConfig:
        if self.log_every_iters != 1:
            raise ValueError(
                "trainer.log_every_iters is not implemented as a metric-sink "
                "throttle; keep it at 1 so every iteration is recorded"
            )
        return self
