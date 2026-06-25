"""Versioned source of truth: frozen datasets, their schema, and run lock."""

from pathlib import Path

DIR = Path(__file__).resolve().parent
BASELINES = DIR / "baselines.parquet"
SWEEPS = DIR / "sweeps.parquet"
RUNS_LOCK = DIR / "runs.lock.json"
