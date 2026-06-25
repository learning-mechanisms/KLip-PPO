"""Fetch histories and summaries for the pinned runs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def histories(api: Any, lock: dict[str, Any]) -> pd.DataFrame:
    """Per-iteration metrics for every pinned baseline run."""
    columns = lock["metrics"]["core"] + lock["metrics"]["beta"]
    project = lock["project"]
    frames = []
    for entry in lock["baselines"]:
        run = api.run(f"{project}/{entry['id']}")
        history = run.history(samples=2000, pandas=True)
        history = history.reindex(columns=columns)
        history["env"] = entry["env"]
        history["algo"] = entry["algo"]
        history["seed"] = entry["seed"]
        frames.append(history)
    return pd.concat(frames, ignore_index=True)


def summaries(api: Any, lock: dict[str, Any]) -> pd.DataFrame:
    """Final return and pinned knob settings for every pinned sweep run."""
    project = lock["project"]
    rows = []
    for entry in lock["sweeps"]:
        run = api.run(f"{project}/{entry['id']}")
        row = {key: value for key, value in entry.items() if key != "id"}
        row["final_return"] = run.summary.get("train/return/mean")
        rows.append(row)
    return pd.DataFrame(rows)
