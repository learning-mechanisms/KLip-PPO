"""Single entry point for reading the frozen datasets."""

from __future__ import annotations

import pandas as pd

from analysis.datasets import BASELINES, SWEEPS, schema


def baselines() -> pd.DataFrame:
    return schema.check_baselines(pd.read_parquet(BASELINES))


def sweeps() -> pd.DataFrame:
    return schema.check_sweeps(pd.read_parquet(SWEEPS))
