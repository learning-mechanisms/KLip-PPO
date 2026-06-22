"""Render the paper figures from the frozen baselines and sweeps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.render import derive, style

CLIP = "#1f77b4"
PER_SAMPLE = "#c0392b"
PASS = "#1f77b4"
KILL = "#c0392b"


def equivalence(
    df: pd.DataFrame, out_dir: Path, *, envs: list[str] | None = None, ext: str = "pdf"
) -> list[Path]:
    """One panel per MuJoCo task: all four variants with seed bands."""
    paths = []
    for env in envs or derive.MUJOCO:
        fig, ax = plt.subplots(figsize=(3.7, 2.9))
        for algo, label, kw in style.VARIANTS:
            mean, std = derive.curve(df, env, algo)
            ax.plot(mean.index, mean.values, label=label, **kw)
            if algo != "ppo_kl_per_sample":
                ax.fill_between(
                    mean.index,
                    mean - std,
                    mean + std,
                    color=kw["color"],
                    alpha=0.12,
                    linewidth=0,
                )
        ax.set_xlabel("environment step")
        ax.set_ylabel("episode return")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="xx-small", loc="lower right")
        fig.tight_layout()
        paths.append(
            style.save(fig, out_dir / f"equiv_{derive.short(env).lower()}.{ext}")
        )
    return paths


def identity(
    df: pd.DataFrame, out_dir: Path, *, envs: list[str] | None = None, ext: str = "pdf"
) -> list[Path]:
    """One figure per task: PPO-Clip and per-sample KL on shared-y panels."""
    pair = [
        ("ppo_clip", "PPO-Clip", CLIP),
        ("ppo_kl_per_sample", "per-sample KL", PER_SAMPLE),
    ]
    paths = []
    for env in envs or derive.ALL_TASKS:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
        for ax, (algo, label, color) in zip(axes, pair):
            mean, std = derive.curve(df, env, algo)
            ax.plot(mean.index, mean.values, color=color, lw=1.5)
            ax.fill_between(
                mean.index,
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
            ax.set_title(label)
            ax.set_xlabel("environment step")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("episode return")
        fig.tight_layout()
        paths.append(
            style.save(fig, out_dir / f"identity_{derive.short(env).lower()}.{ext}")
        )
    return paths


def partition(
    df: pd.DataFrame, out_dir: Path, *, envs: list[str] | None = None, ext: str = "pdf"
) -> list[Path]:
    """One figure per MuJoCo task: clip-batch fraction in the kill and pass sets."""
    series = [
        ("i_kill", r"$\mathcal{I}_{\mathrm{kill}}$", KILL),
        ("i_pass", r"$\mathcal{I}_{\mathrm{pass}}$", PASS),
    ]
    paths = []
    for env in envs or derive.MUJOCO:
        fig, ax = plt.subplots(figsize=(3.7, 2.9))
        for region, label, color in series:
            mean, std = derive.partition(df, env, region)
            ax.plot(mean.index, mean.values, label=label, color=color, lw=1.5)
            ax.fill_between(
                mean.index, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )
        ax.set_xlabel("environment step")
        ax.set_ylabel("batch fraction")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="x-small", loc="upper right")
        fig.tight_layout()
        paths.append(
            style.save(fig, out_dir / f"partition_{derive.short(env).lower()}.{ext}")
        )
    return paths


def beta(
    df: pd.DataFrame, out_dir: Path, *, envs: list[str] | None = None, ext: str = "pdf"
) -> list[Path]:
    """One figure per MuJoCo task: per-sample coefficient with percentile bands."""
    paths = []
    for env in envs or derive.MUJOCO:
        q = derive.beta_quantiles(df, env)
        fig, ax = plt.subplots(figsize=(3.7, 2.9))
        ax.plot(q.index, q["beta_p50"], color=PER_SAMPLE, lw=1.5)
        ax.fill_between(
            q.index,
            q["beta_p10"],
            q["beta_p90"],
            color=PER_SAMPLE,
            alpha=0.25,
            linewidth=0,
        )
        ax.fill_between(
            q.index,
            q["beta_p01"],
            q["beta_p99"],
            color=PER_SAMPLE,
            alpha=0.12,
            linewidth=0,
        )
        ax.set_xlabel("environment step")
        ax.set_ylabel(r"$\beta_t$")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        paths.append(
            style.save(fig, out_dir / f"beta_{derive.short(env).lower()}.{ext}")
        )
    return paths


def sweep_knobs(df: pd.DataFrame, out_dir: Path) -> Path:
    """Final return vs each trust-region knob, on a log knob axis."""
    specs = [
        ("clip_epsilon", "ppo_clip", r"clip $\epsilon$"),
        ("beta", "ppo_kl_fixed", r"fixed $\beta$"),
        ("kl_target", "ppo_kl_adaptive", r"adaptive KL target"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (knob, algo, label) in zip(axes, specs):
        for env in derive.SWEEP_TASKS:
            x, mean, sem = derive.sweep(df, algo, knob, env)
            ax.errorbar(
                x, mean, yerr=sem, marker="o", capsize=3, label=derive.short(env)
            )
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize="x-small")
    axes[0].set_ylabel("final return")
    fig.tight_layout()
    return style.save_pdf(fig, out_dir / "sweep_knobs.pdf")
