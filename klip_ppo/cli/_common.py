"""Shared helpers for the CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from klip_ppo.configs.experiment import ExperimentConfig, apply_overrides, load_yaml


def load_experiment_from_yaml(
    path: Path,
    *,
    overrides: list[str] | None = None,
    seed: int | None = None,
    name: str | None = None,
) -> ExperimentConfig:
    data: dict[str, Any] = load_yaml(path)
    if overrides:
        data = apply_overrides(data, overrides)
    if seed is not None:
        data["seed"] = int(seed)
    if name is not None:
        data["name"] = name
    return ExperimentConfig.model_validate(data)


def load_experiment_from_snapshot(
    snapshot_path: Path,
    *,
    overrides: list[str] | None = None,
    seed: int | None = None,
    name: str | None = None,
) -> ExperimentConfig:
    snapshot = json.loads(snapshot_path.read_text())
    if isinstance(snapshot, dict) and "config" in snapshot:
        data = dict(snapshot["config"])
    else:
        data = dict(snapshot)
    if overrides:
        data = apply_overrides(data, overrides)
    if seed is not None:
        data["seed"] = int(seed)
    if name is not None:
        data["name"] = name
    return ExperimentConfig.model_validate(data)
