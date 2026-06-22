"""Stage, validate, and zip the reproducible figure artifact."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = Path(__file__).resolve().parent / "dist"
STAGE = DIST / "artifact"
ARCHIVE = DIST / "artifact.zip"
