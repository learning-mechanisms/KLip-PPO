"""Default WandB identity helpers for train runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.utils.ids import slugify
from klip_ppo.utils.paths import PRESETS_DIR, SNAPSHOTS_DIR


def default_wandb_group(cfg: ExperimentConfig) -> str:
    """Group seed replicas by task, algorithm, and experiment setting."""
    return "__".join(
        (
            slugify(cfg.env.id),
            slugify(cfg.algorithm.kind),
            slugify(cfg.name),
        )
    )


def source_wandb_identity(source_path: Path) -> str | None:
    """Return a WandB identity aligned with a preset snapshot path/envelope."""
    if source_path.suffix.lower() == ".json":
        identity = _identity_from_snapshot_json(source_path)
        if identity is not None:
            return identity
        return _identity_from_path(source_path, root=SNAPSHOTS_DIR)

    if source_path.suffix.lower() in {".yaml", ".yml"}:
        matching_snapshot = _find_matching_snapshot(source_path)
        if matching_snapshot is not None:
            return source_wandb_identity(matching_snapshot)
        return _identity_from_path(source_path, root=PRESETS_DIR)

    return _identity_from_path(source_path)


def wandb_group(cfg: ExperimentConfig, *, source_identity: str | None = None) -> str:
    """Return the effective WandB group for a run config."""
    if cfg.logging.wandb is not None and cfg.logging.wandb.group is not None:
        return cfg.logging.wandb.group
    if source_identity is not None:
        return source_identity
    return default_wandb_group(cfg)


def wandb_run_name(
    cfg: ExperimentConfig,
    *,
    seed: int,
    run_dir: Path,
    source_identity: str | None = None,
) -> str:
    """Return the effective WandB run name, including the seed replica."""
    wandb_cfg = cfg.logging.wandb
    if wandb_cfg is None:
        raise ValueError("wandb_run_name requires cfg.logging.wandb to be set")
    base = wandb_cfg.run_name or wandb_group(cfg, source_identity=source_identity)
    return f"{base}__seed={seed}__{run_dir.name}"


def _identity_from_snapshot_json(path: Path) -> str | None:
    try:
        snapshot: Any = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(snapshot, dict):
        return None
    group = snapshot.get("group")
    name = snapshot.get("name")
    if isinstance(group, str) and isinstance(name, str):
        return _join_identity(group, name)
    return None


def _find_matching_snapshot(source_path: Path) -> Path | None:
    root = SNAPSHOTS_DIR / "presets"
    if not root.exists():
        return None
    matches = sorted(root.rglob(f"{source_path.stem}.json"))
    if len(matches) == 1:
        return matches[0]
    return None


def _identity_from_path(path: Path, *, root: Path | None = None) -> str:
    path = path.resolve()
    try:
        rel = path.relative_to((root or path.parent).resolve())
    except ValueError:
        return _join_identity(path.stem)
    parts = (*rel.parent.parts, rel.stem)
    return _join_identity(*(part for part in parts if part))


def _join_identity(*parts: str) -> str:
    return "__".join(slugify(part) for part in parts if part)
