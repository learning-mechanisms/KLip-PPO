"""
Filesystem locations.

Single source of truth. ``PROJECT_ROOT`` is resolved at import time by walking up from
this file until a ``pixi.toml`` is found. Nothing else in the codebase should hard-code
a path.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path


@cache
def project_root() -> Path:
    """Return the repository root (directory containing ``pixi.toml``)."""
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pixi.toml").exists():
            return candidate
    raise RuntimeError(
        f"could not locate project root: no pixi.toml found above {here}"
    )


PROJECT_ROOT: Path = project_root()
ARTIFACTS_DIR: Path = Path(
    os.environ.get("KLIP_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
).resolve()
RUNS_DIR: Path = ARTIFACTS_DIR / "runs"
SWEEPS_DIR: Path = ARTIFACTS_DIR / "sweeps"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
PRESETS_DIR: Path = CONFIGS_DIR / "presets"
SNAPSHOTS_DIR: Path = CONFIGS_DIR / "snapshots"
PIXI_LOCK: Path = PROJECT_ROOT / "pixi.lock"
