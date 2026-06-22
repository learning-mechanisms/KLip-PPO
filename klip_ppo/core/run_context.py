"""Immutable per-Job execution context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass(frozen=True)
class RunContext:
    """
    What every component in a Job receives.

    Holds the device, seed, run directory, and the reproducibility
    fingerprint of the launching environment. There is no concept of
    rank or world_size: a Job is single-device by construction.
    """

    device: torch.device
    seed: int
    run_dir: Path
    git_commit: str
    git_dirty: bool
    pixi_lock_sha: str | None
    started_at: datetime
