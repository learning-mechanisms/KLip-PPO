"""Report the staged artifact size and bound it."""

from __future__ import annotations

from pathlib import Path

from submission import STAGE

MAX_MB = 250


def _total_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def report() -> None:
    size = _total_bytes(STAGE)
    mb = size / 1e6
    print(f"artifact: {mb:.1f} MB ({size} bytes)")
    if mb > MAX_MB:
        raise ValueError(f"artifact {mb:.1f} MB exceeds {MAX_MB} MB")
