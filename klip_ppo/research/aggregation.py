"""Load run artifacts into a tidy DataFrame for plotting / reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_runs(
    runs_root: Path,
    *,
    algos: list[str] | None = None,
    env: str | None = None,
) -> pd.DataFrame:
    """
    Concatenate every ``train.parquet`` under ``runs_root``.

    Adds ``algo``, ``env``, ``seed``, ``experiment``, ``run_dir`` columns derived from
    the canonical run-dir layout.
    """
    frames: list[pd.DataFrame] = []
    for snapshot_path in sorted(runs_root.rglob("snapshot.json")):
        run_dir = snapshot_path.parent
        parts = _decode_run_dir(run_dir, runs_root)
        if parts is None:
            continue
        experiment, algo, env_id, seed = parts
        if algos and algo not in algos:
            continue
        if env is not None and env_id != env:
            continue
        parquet_path = run_dir / "metrics" / "train.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        df["experiment"] = experiment
        df["algo"] = algo
        df["env"] = env_id
        df["seed"] = seed
        df["run_dir"] = str(run_dir)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_snapshot(run_dir: Path) -> dict:
    return json.loads((run_dir / "snapshot.json").read_text())


def _decode_run_dir(run_dir: Path, runs_root: Path) -> tuple[str, str, str, int] | None:
    try:
        rel = run_dir.relative_to(runs_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) != 5:
        return None
    experiment, algo, env_id, seed_dir, _leaf = parts
    if not seed_dir.startswith("seed="):
        return None
    try:
        seed = int(seed_dir.split("=", 1)[1])
    except ValueError:
        return None
    return experiment, algo, env_id, seed
