"""Shared names for PPO diagnostic metrics."""

from __future__ import annotations

BETA_QUANTILES: tuple[tuple[str, float], ...] = (
    ("p01", 0.01),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)

BETA_SAMPLE_DIAGNOSTIC_PREFIXES: tuple[str, ...] = (
    "beta/per_sample/all",
    "beta/per_sample/I_kill",
    "beta/times_adv_abs",
)

BETA_QUANTILE_KEYS: tuple[str, ...] = tuple(
    f"{prefix}/{label}"
    for prefix in BETA_SAMPLE_DIAGNOSTIC_PREFIXES
    for label, _ in BETA_QUANTILES
)
