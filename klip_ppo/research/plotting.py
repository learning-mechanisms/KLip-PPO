"""Matplotlib plot builders consuming the parquet metric tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

DEFAULT_PLOT_FORMAT = "pdf"
DEFAULT_PLOT_SUFFIX = f".{DEFAULT_PLOT_FORMAT}"
RASTER_PLOT_DPI = 120


def plot_learning_curves(df: pd.DataFrame, out: Path) -> Path:
    """Median + IQR learning curves: one subplot per env, one line per algo."""
    envs = sorted(df["env"].unique())
    algos = sorted(df["algo"].unique())
    if not envs:
        raise ValueError("no envs to plot")
    cols = min(3, len(envs))
    rows = (len(envs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    for ax, env in zip(axes.flat, envs, strict=False):
        env_df = df[df["env"] == env]
        for algo in algos:
            algo_df = env_df[env_df["algo"] == algo]
            if algo_df.empty:
                continue
            grouped = algo_df.groupby("time/env_step")["train/return/mean"]
            median = grouped.median()
            lo = grouped.quantile(0.25)
            hi = grouped.quantile(0.75)
            ax.plot(median.index, median.values, label=algo)
            ax.fill_between(median.index, lo.values, hi.values, alpha=0.15)
        ax.set_title(env)
        ax.set_xlabel("env step")
        ax.set_ylabel("episode return")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize="x-small")
    for ax in list(axes.flat)[len(envs) :]:
        ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, out)
    plt.close(fig)
    return out


def plot_beta_quantile_band(
    df: pd.DataFrame,
    out: Path,
    *,
    quantity: str = "beta/per_sample/all",
) -> Path:
    """
    Per-sample β_t quantile bands over training (Phase B figure 4).

    For each (env, algo), plots the median (p50) line with a p10–p90 inner band and a
    p01–p99 outer band. Across-seed aggregation is the median of each per-rollout
    quantile column.

    ``quantity`` is one of ``beta/per_sample/all``, ``beta/per_sample/I_kill``,
    ``beta/times_adv_abs``.
    """
    p50_col = f"{quantity}/p50"
    p10_col = f"{quantity}/p10"
    p90_col = f"{quantity}/p90"
    p01_col = f"{quantity}/p01"
    p99_col = f"{quantity}/p99"
    required = (p50_col, p10_col, p90_col, p01_col, p99_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing quantile columns for {quantity!r}: {missing}")
    envs = sorted(df["env"].unique())
    algos = sorted(df["algo"].unique())
    if not envs:
        raise ValueError("no envs to plot")
    cols = min(3, len(envs))
    rows = (len(envs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    for ax, env in zip(axes.flat, envs, strict=False):
        env_df = df[df["env"] == env]
        for algo in algos:
            algo_df = env_df[env_df["algo"] == algo]
            if algo_df.empty or algo_df[p50_col].isna().all():
                continue
            grouped = algo_df.groupby("time/env_step")
            median = grouped[p50_col].median()
            inner_lo = grouped[p10_col].median()
            inner_hi = grouped[p90_col].median()
            outer_lo = grouped[p01_col].median()
            outer_hi = grouped[p99_col].median()
            (line,) = ax.plot(median.index, median.values, label=algo)
            color = line.get_color()
            ax.fill_between(
                median.index, inner_lo.values, inner_hi.values, alpha=0.25, color=color
            )
            ax.fill_between(
                median.index, outer_lo.values, outer_hi.values, alpha=0.10, color=color
            )
        ax.set_title(env)
        ax.set_xlabel("env step")
        ax.set_ylabel(quantity)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize="x-small")
    for ax in list(axes.flat)[len(envs) :]:
        ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, out)
    plt.close(fig)
    return out


def plot_kl_vs_clip(run_dir: Path, out: Path) -> Path:
    """Per-run policy KL, clip fraction, and kill-partition occupancy over time."""
    parquet = run_dir / "metrics" / "train.parquet"
    df = pd.read_parquet(parquet)
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for ax, key, title in (
        (axes[0], "policy/kl/approx", "approximate KL"),
        (axes[1], "policy/clip/fraction", "clip fraction"),
        (axes[2], "policy/partition/I_kill/fraction", "frac in I_kill"),
    ):
        if key in df.columns and df[key].notna().any():
            ax.plot(df["time/env_step"], df[key])
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("env step")
    fig.tight_layout()
    _save_figure(fig, out)
    plt.close(fig)
    return out


def plot_final_returns_table(df: pd.DataFrame, out: Path) -> Path:
    """Final mean ± stderr return table (env × algo)."""
    if df.empty:
        raise ValueError("no rows to summarise")
    last = df.sort_values("time/env_step").groupby(["env", "algo", "seed"]).tail(1)
    stats = (
        last.groupby(["env", "algo"])["train/return/mean"]
        .agg(["mean", _stderr])
        .reset_index()
        .rename(columns={"_stderr": "stderr"})
    )
    fig, ax = plt.subplots(
        figsize=(2 + 1.2 * len(stats["algo"].unique()), 0.5 + 0.4 * len(stats))
    )
    ax.axis("off")
    cell_text = [
        [
            row["env"],
            row["algo"],
            f"{row['mean']:.2f} ± {row['stderr']:.2f}",
        ]
        for _, row in stats.iterrows()
    ]
    ax.table(
        cellText=cell_text,
        colLabels=["env", "algo", "final return"],
        loc="center",
        cellLoc="left",
    )
    fig.tight_layout()
    _save_figure(fig, out)
    plt.close(fig)
    return out


def _save_figure(fig: Figure, out: Path) -> None:
    plot_format = out.suffix.removeprefix(".").lower() or DEFAULT_PLOT_FORMAT

    with plt.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        if plot_format in {"png", "jpg", "jpeg", "tif", "tiff", "webp"}:
            fig.savefig(out, format=plot_format, dpi=RASTER_PLOT_DPI)
        else:
            fig.savefig(out, format=plot_format)


def _stderr(values: pd.Series) -> float:
    n = values.count()
    if n < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(n))
