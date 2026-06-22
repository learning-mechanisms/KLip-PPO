"""Map raw run metrics to the dataset schema and write deterministically."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.datasets import schema

BASELINE_RENAME = {
    "time/env_step": "env_step",
    "train/return/mean": "return_mean",
    "policy/kl/approx": "kl_approx",
    "policy/clip/fraction": "clip_fraction",
    "policy/partition/I_in/fraction": "i_in",
    "policy/partition/I_kill/fraction": "i_kill",
    "policy/partition/I_pass/fraction": "i_pass",
    "beta/per_sample/all/p01": "beta_p01",
    "beta/per_sample/all/p10": "beta_p10",
    "beta/per_sample/all/p50": "beta_p50",
    "beta/per_sample/all/p90": "beta_p90",
    "beta/per_sample/all/p99": "beta_p99",
}

BASELINE_SORT = ["env", "algo", "seed", "env_step"]
SWEEP_SORT = ["env", "algo", "clip_epsilon", "beta", "kl_target", "seed"]


def normalize_baselines(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=BASELINE_RENAME)[list(schema.BASELINES)]
    df = df.astype(schema.BASELINES)
    df = df.sort_values(BASELINE_SORT).reset_index(drop=True)
    return schema.check_baselines(df)


def normalize_sweeps(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[list(schema.SWEEPS)].astype(schema.SWEEPS)
    df = df.sort_values(SWEEP_SORT, na_position="last").reset_index(drop=True)
    return schema.check_sweeps(df)


def write(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
