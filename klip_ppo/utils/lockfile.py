"""Pixi lockfile hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from klip_ppo.utils.paths import PIXI_LOCK


def pixi_lock_sha256(path: Path = PIXI_LOCK) -> str | None:
    """Return the sha256 of the pixi lockfile, or ``None`` if it is missing."""
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
