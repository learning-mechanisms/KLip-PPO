"""Tabular summaries (final returns, mechanism stats)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def final_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ``(env × algo)`` table of final-return mean and standard error.

    Prefers periodic ``eval/return/mean`` (raw environment return, deterministic policy,
    no reward normalisation) when present, falling back to ``train/return/mean``
    (training-rollout return) otherwise. The paper's headline curves still use training-
    rollout returns; eval is the literature-comparable final-performance signal.
    """
    if df.empty:
        return df
    last = df.sort_values("time/env_step").groupby(["env", "algo", "seed"]).tail(1)
    if "eval/return/mean" in df.columns and last["eval/return/mean"].notna().any():
        return_col = "eval/return/mean"
        last = last[last[return_col].notna()]
    else:
        return_col = "train/return/mean"
    return (
        last.groupby(["env", "algo"])[return_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .assign(
            stderr=lambda d: d["std"] / np.sqrt(d["count"]),
            return_source=return_col,
        )
    )


def partition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean partition occupancy over training, by ``env × algo``."""
    cols = [
        "policy/partition/I_in/fraction",
        "policy/partition/I_pass/fraction",
        "policy/partition/I_kill/fraction",
    ]
    present = [c for c in cols if c in df.columns]
    if not present or df.empty:
        return pd.DataFrame()
    return df.groupby(["env", "algo"])[present].mean().reset_index()
