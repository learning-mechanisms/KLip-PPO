"""Deterministic matplotlib setup and figure writer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Pin the PDF timestamp source so output is byte-stable across invocations.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "klip-ppo",
    "figure.dpi": 150,
    "font.size": 9,
    "axes.grid": False,
}

# Per-variant colour and line style, shared by figures and the site.
VARIANTS: list[tuple[str, str, dict[str, Any]]] = [
    ("ppo_clip", "PPO-Clip", {"color": "#1f77b4", "lw": 2.0, "ls": "-"}),
    ("ppo_kl_per_sample", "per-sample KL", {"color": "#000000", "lw": 1.3, "ls": "--"}),
    ("ppo_kl_fixed", r"fixed $\beta$", {"color": "#cf6a1a", "lw": 1.5, "ls": "-"}),
    (
        "ppo_kl_adaptive",
        r"adaptive $\beta$",
        {"color": "#1f7a4d", "lw": 1.5, "ls": "-"},
    ),
]


def save_pdf(fig: plt.Figure, path: Path) -> Path:
    """Write a PDF without an embedded timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig.savefig(
            path, format="pdf", bbox_inches="tight", metadata={"CreationDate": None}
        )
    plt.close(fig)
    return path


def save_svg(fig: plt.Figure, path: Path) -> Path:
    """Write a vector SVG without an embedded timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return path


def save(fig: plt.Figure, path: Path) -> Path:
    """Write ``fig`` to ``path``, choosing the format from its suffix."""
    return save_svg(fig, path) if path.suffix == ".svg" else save_pdf(fig, path)
