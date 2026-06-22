"""Assemble a self-contained copy of the code, data, and pinned environment."""

from __future__ import annotations

import shutil
from pathlib import Path

from submission import ROOT, STAGE

INCLUDE = ["analysis", "pixi.toml", "pixi.lock", "pyproject.toml"]

RUN = """\
#!/usr/bin/env bash
set -euo pipefail
pixi install
pixi run python -m analysis.render --out figures
"""

README = """\
# Reproducible figures

```bash
./run.sh
```

Reads the frozen datasets in `analysis/datasets/`, rebuilds every figure into
`figures/`, and matches `analysis/figures.sha256`. No network access.
"""


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for name in INCLUDE:
        src = ROOT / name
        dst = STAGE / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)
    (STAGE / "run.sh").write_text(RUN)
    (STAGE / "run.sh").chmod(0o755)
    (STAGE / "README.md").write_text(README)
    return STAGE
