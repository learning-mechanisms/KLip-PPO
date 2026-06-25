"""Write a byte-stable zip of the staged artifact."""

from __future__ import annotations

import zipfile
from pathlib import Path

from submission import ARCHIVE, STAGE

FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def write() -> Path:
    files = sorted(p for p in STAGE.rglob("*") if p.is_file())
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(str(path.relative_to(STAGE)), date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0o777) << 16
            archive.writestr(info, path.read_bytes())
    print(f"archive: {ARCHIVE}")
    return ARCHIVE
