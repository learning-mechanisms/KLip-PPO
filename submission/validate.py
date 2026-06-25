"""Reproducibility checks: offline code, schema, and figure checksums."""

from __future__ import annotations

import tempfile
from pathlib import Path

from analysis.render import __main__ as build
from analysis.render import checksums, sources

from submission import ROOT

OFFLINE = [ROOT / "analysis" / "render", ROOT / "analysis" / "datasets"]


def _offline(roots: list[Path]) -> None:
    hits = [
        str(p)
        for root in roots
        for p in root.rglob("*.py")
        if "import wandb" in p.read_text()
    ]
    if hits:
        raise ValueError(f"online import in offline code: {hits}")


def validate() -> None:
    _offline(OFFLINE)
    sources.baselines()
    sources.sweeps()
    with tempfile.TemporaryDirectory() as tmp:
        build.build(Path(tmp))
        checksums.verify(Path(tmp))
    print("validate: OK")
