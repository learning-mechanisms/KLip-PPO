"""Aggregations over the frozen datasets."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from analysis.export.runs import KNOB_COLUMNS, manifest

TAIL_FRACTION = 0.1


class IncompleteDataError(RuntimeError):
    """Raised when a figure's inputs are missing expected runs."""


def _tasks(category: str, suite: str | None = None) -> list[str]:
    envs = {
        cell["env"]
        for cell in manifest()
        if cell["category"] == category and (suite is None or cell["suite"] == suite)
    }
    return sorted(envs)


MUJOCO = _tasks("baselines", "mujoco")
ALL_TASKS = _tasks("baselines")
SWEEP_TASKS = _tasks("sweeps")


def short(env: str) -> str:
    return env.split("-v")[0]


def _knob_key(values: Iterable[float | None]) -> tuple:
    return tuple(
        None if value is None or pd.isna(value) else float(value) for value in values
    )


def require_complete(df: pd.DataFrame, category: str) -> None:
    """Raise unless every run expected by the manifest is present for ``category``."""
    missing = []
    for cell in manifest():
        if cell["category"] != category:
            continue
        sub = df[(df["env"] == cell["env"]) & (df["algo"] == cell["algo"])]
        if category == "sweeps":
            target = _knob_key(cell["knobs"][column] for column in KNOB_COLUMNS)
            present = {
                int(row["seed"])
                for _, row in sub.iterrows()
                if _knob_key(row[column] for column in KNOB_COLUMNS) == target
                and pd.notna(row["final_return"])
            }
        else:
            present = {int(seed) for seed in sub["seed"].unique()}
        missing += [
            f"{cell['identity']} seed={seed}"
            for seed in cell["seeds"]
            if seed not in present
        ]
    if missing:
        listing = "\n".join(f"  {item}" for item in sorted(missing))
        raise IncompleteDataError(
            f"{category}: {len(missing)} expected runs absent:\n{listing}"
        )


def curve(df: pd.DataFrame, env: str, algo: str) -> tuple[pd.Series, pd.Series]:
    """Mean and population std across seeds, indexed by env_step."""
    g = df[(df.env == env) & (df.algo == algo)].groupby("env_step")["return_mean"]
    return g.mean(), g.std(ddof=0)


def final_return(df: pd.DataFrame, env: str, algo: str) -> tuple[float, float]:
    """Tail-window mean and std across seeds."""
    sub = df[(df.env == env) & (df.algo == algo)]
    threshold = sub["env_step"].quantile(1.0 - TAIL_FRACTION)
    per_seed = sub[sub["env_step"] >= threshold].groupby("seed")["return_mean"].mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=0))


def partition(df: pd.DataFrame, env: str, region: str) -> tuple[pd.Series, pd.Series]:
    """Clip-batch fraction in a region, mean and std across seeds by step."""
    g = df[(df.env == env) & (df.algo == "ppo_clip")].groupby("env_step")[region]
    return g.mean(), g.std(ddof=0)


def beta_quantiles(df: pd.DataFrame, env: str) -> pd.DataFrame:
    """Median over seeds of each per-sample beta percentile, indexed by step."""
    cols = ["beta_p01", "beta_p10", "beta_p50", "beta_p90", "beta_p99"]
    sub = df[(df.env == env) & (df.algo == "ppo_kl_per_sample")]
    return sub.groupby("env_step")[cols].median()


def sweep(df: pd.DataFrame, algo: str, knob: str, env: str):
    """Final return vs knob value: mean and standard error over seeds."""
    sub = df[(df.algo == algo) & df[knob].notna() & (df.env == env)]
    stat = sub.groupby(knob)["final_return"].agg(["mean", "std", "count"]).sort_index()
    sem = stat["std"] / np.sqrt(stat["count"].clip(lower=1))
    return stat.index.to_numpy(dtype=float), stat["mean"].to_numpy(), sem.to_numpy()
