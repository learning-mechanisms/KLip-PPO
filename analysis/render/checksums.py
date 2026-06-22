"""Checksum manifest of the built figures and table."""

from __future__ import annotations

import hashlib
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "figures.sha256"


def manifest(out_dir: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out_dir.iterdir())
        if path.is_file()
    }


def write(out_dir: Path, path: Path = MANIFEST) -> None:
    lines = [f"{digest}  {name}" for name, digest in manifest(out_dir).items()]
    path.write_text("\n".join(lines) + "\n")


def verify(out_dir: Path, path: Path = MANIFEST) -> None:
    expected = {}
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    actual = manifest(out_dir)
    if actual != expected:
        diff = sorted(set(expected) ^ set(actual)) or [
            name for name in actual if actual[name] != expected.get(name)
        ]
        raise ValueError(f"figure checksums differ: {diff}")
