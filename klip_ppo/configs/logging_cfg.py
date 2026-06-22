"""Logging configuration (stdout, parquet, wandb, tensorboard)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from klip_ppo.configs.base import BaseConfig

WandbMode = Literal["online", "offline", "disabled", "shared"]


class WandbConfig(BaseConfig):
    """Weights & Biases sink configuration."""

    project: str
    entity: str | None = None
    group: str | None = None
    """WandB group name shared by related runs."""

    tags: tuple[str, ...] = ()
    mode: WandbMode = "online"
    """WandB connection mode for run logging."""

    run_name: str | None = None
    """Explicit WandB run name; generated when omitted."""

    job_type: str = "train"
    upload_artifacts: bool = True
    """Whether to upload final run artifacts to WandB."""

    artifact_aliases: tuple[str, ...] = ("latest",)
    """Aliases assigned to uploaded WandB artifacts."""

    resume: Literal["allow", "never", "must"] = Field(
        default="never",
        description="WandB resume policy for matching run identifiers.",
    )


class LoggingConfig(BaseConfig):
    """Where training metrics are written."""

    stdout: bool = True
    parquet: bool = True
    """Whether to write local parquet metric files."""

    tensorboard: bool = False
    """Whether to write TensorBoard event files."""

    wandb: WandbConfig | None = None
    """Optional WandB sink configuration."""

    parquet_flush_every_iters: int = Field(default=10, gt=0)
    """Metric-row interval used to flush parquet writers."""
