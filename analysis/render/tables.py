"""Render the final-return LaTeX table from the frozen baselines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.render import derive

COLUMNS = ["ppo_clip", "ppo_kl_per_sample", "ppo_kl_fixed", "ppo_kl_adaptive"]
LABELS = {
    "ppo_clip": "PPO-Clip",
    "ppo_kl_per_sample": "Per-sample",
    "ppo_kl_fixed": r"Fixed $\beta$",
    "ppo_kl_adaptive": r"Adaptive $\beta$",
}


def final_returns(df: pd.DataFrame, out_dir: Path) -> Path:
    rows = [
        r"\begin{tabular}{l rrrr}",
        r"\toprule",
        "Task & " + " & ".join(LABELS[c] for c in COLUMNS) + r" \\",
        r"\midrule",
    ]
    for env in derive.ALL_TASKS:
        cells = []
        for algo in COLUMNS:
            mean, std = derive.final_return(df, env, algo)
            cells.append(f"${mean:.0f} \\pm {std:.0f}$")
        rows.append(f"{derive.short(env)} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "final_returns.tex"
    out.write_text("\n".join(rows) + "\n")
    return out
