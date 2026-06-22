"""Environment helpers for opt-in Weights & Biases logging."""

from __future__ import annotations

import os
from typing import cast

from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.logging_cfg import WandbConfig, WandbMode


def with_wandb_from_env(
    cfg: ExperimentConfig,
    *,
    project: str | None = None,
    entity: str | None = None,
    mode: WandbMode | str | None = None,
) -> ExperimentConfig:
    """
    Enable WandB logging from environment-style values when unset.

    Args:
        cfg: Experiment configuration to update.
        project: Explicit WandB project, falling back to ``WANDB_PROJECT``.
        entity: Explicit WandB entity, falling back to ``WANDB_ENTITY``.
        mode: Explicit WandB mode, falling back to ``WANDB_MODE``.

    Returns:
        The original config when WandB is already configured or no project is set;
        otherwise a copy with ``logging.wandb`` populated.
    """
    if cfg.logging.wandb is not None:
        return cfg

    wandb_project = project or os.environ.get("WANDB_PROJECT")
    if not wandb_project:
        return cfg

    wandb_mode = cast(WandbMode, mode or os.environ.get("WANDB_MODE") or "online")
    wandb_cfg = WandbConfig(
        project=wandb_project,
        entity=entity or os.environ.get("WANDB_ENTITY") or None,
        mode=wandb_mode,
    )
    return cfg.model_copy(
        update={"logging": cfg.logging.model_copy(update={"wandb": wandb_cfg})}
    )
