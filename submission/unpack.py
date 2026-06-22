"""Extract the archive into a clean dir and rebuild the figures."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from analysis.render import checksums

from submission import ARCHIVE


def unpack_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(root)
        figures = root / "figures"
        subprocess.run(
            [sys.executable, "-m", "analysis.render", "--out", str(figures)],
            cwd=root,
            check=True,
        )
        checksums.verify(figures, root / "analysis" / "figures.sha256")
    print("unpack: OK")
