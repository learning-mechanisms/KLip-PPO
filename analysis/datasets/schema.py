"""Column and dtype contracts for the frozen datasets."""

from __future__ import annotations

import pandas as pd

# Per-iteration history, one row per (env, algo, seed, env_step).
BASELINES: dict[str, str] = {
    "env": "object",
    "algo": "object",
    "seed": "int64",
    "env_step": "int64",
    "return_mean": "float64",
    "kl_approx": "float64",
    "clip_fraction": "float64",
    "i_in": "float64",
    "i_kill": "float64",
    "i_pass": "float64",
    "beta_p01": "float64",
    "beta_p10": "float64",
    "beta_p50": "float64",
    "beta_p90": "float64",
    "beta_p99": "float64",
}

# Final return per swept trust-region knob; unused knob columns are NaN.
SWEEPS: dict[str, str] = {
    "env": "object",
    "algo": "object",
    "seed": "int64",
    "final_return": "float64",
    "clip_epsilon": "float64",
    "beta": "float64",
    "kl_target": "float64",
}


def _check(df: pd.DataFrame, spec: dict[str, str], name: str) -> pd.DataFrame:
    if list(df.columns) != list(spec):
        raise ValueError(f"{name}: columns {list(df.columns)} != {list(spec)}")
    return df.astype(spec)


def check_baselines(df: pd.DataFrame) -> pd.DataFrame:
    return _check(df, BASELINES, "baselines")


def check_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    return _check(df, SWEEPS, "sweeps")
