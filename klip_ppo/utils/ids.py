"""Stable, filesystem-safe identifiers for runs and sweeps."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(value: str) -> str:
    return _SLUG_RE.sub("-", value).strip("-") or "unnamed"


def utc_timestamp(now: datetime | None = None) -> str:
    """Filesystem-safe UTC timestamp: ``YYYY-MM-DDThh-mm-ssZ``."""
    if now is None:
        now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H-%M-%SZ")


def run_dir(
    *,
    artifacts_root: Path,
    experiment_name: str,
    algo_kind: str,
    env_id: str,
    seed: int,
    timestamp: str,
    git_short: str,
) -> Path:
    """
    Build the canonical run directory path.

    Layout:
        ``<artifacts_root>/runs/<experiment_name>/<algo_kind>/<env_id>/seed=<n>/<ts>__<sha>/``
    """
    leaf = f"{timestamp}__{git_short}"
    return (
        artifacts_root
        / "runs"
        / slugify(experiment_name)
        / slugify(algo_kind)
        / slugify(env_id)
        / f"seed={seed}"
        / leaf
    )
