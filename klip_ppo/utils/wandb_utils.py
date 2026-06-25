"""Small Weights & Biases helpers used by CLIs and loggers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from klip_ppo.configs.logging_cfg import WandbMode

WANDB_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".webp"}


def artifact_name(*parts: str) -> str:
    """Return a WandB-safe, stable artifact name."""
    raw = "-".join(part for part in parts if part)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    return cleaned[:128] or "artifact"


def aliases_or_none(aliases: Iterable[str]) -> list[str] | None:
    cleaned = [alias for alias in aliases if alias]
    return cleaned or None


def publish_file_artifact(
    path: Path,
    *,
    project: str,
    entity: str | None = None,
    run_name: str | None = None,
    mode: WandbMode = "online",
    job_type: str,
    artifact_type: str,
    aliases: Iterable[str] = ("latest",),
    metadata: Mapping[str, Any] | None = None,
    log_image_key: str | None = None,
) -> None:
    """Upload one generated file as a WandB artifact."""
    if mode == "disabled":
        return

    import wandb

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        job_type=job_type,
        config=dict(metadata or {}),
        reinit=True,
    )
    try:
        artifact = wandb.Artifact(
            name=artifact_name(artifact_type, path.parent.name, path.stem),
            type=artifact_type,
            metadata=dict(metadata or {}),
        )
        artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact, aliases=aliases_or_none(aliases))
        if log_image_key is not None and path.suffix.lower() in WANDB_IMAGE_SUFFIXES:
            run.log({log_image_key: wandb.Image(str(path))})
    finally:
        run.finish()


def publish_report_artifact(
    report_path: Path,
    *,
    project: str,
    final_returns,
    partition_stats,
    entity: str | None = None,
    run_name: str | None = None,
    mode: WandbMode = "online",
    aliases: Iterable[str] = ("latest",),
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Upload a markdown report plus aggregate tables to WandB."""
    if mode == "disabled":
        return

    import wandb

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        job_type="report",
        config=dict(metadata or {}),
        reinit=True,
    )
    try:
        artifact = wandb.Artifact(
            name=artifact_name("report", report_path.parent.name, report_path.stem),
            type="report",
            metadata=dict(metadata or {}),
        )
        artifact.add_file(str(report_path), name=report_path.name)
        run.log_artifact(artifact, aliases=aliases_or_none(aliases))

        tables: dict[str, Any] = {}
        if not final_returns.empty:
            tables["report/final_returns"] = wandb.Table(dataframe=final_returns)
        if not partition_stats.empty:
            tables["report/partition_stats"] = wandb.Table(dataframe=partition_stats)
        if tables:
            run.log(tables)
    finally:
        run.finish()
