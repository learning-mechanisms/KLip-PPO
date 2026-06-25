"""Site data and figures generated from the frozen baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.render import derive, figures, style

ROOT = Path(__file__).resolve().parents[2]
WEBSITE = ROOT / "website"

ALGOS = [algo for algo, _label, _kw in style.VARIANTS]
CURVE_POINTS = 80


def _downsample(series: pd.Series, count: int) -> list[list[float]]:
    if len(series) <= count:
        idx: list[int] = list(range(len(series)))
    else:
        idx = np.linspace(0, len(series) - 1, count).round().astype(int).tolist()
    steps = series.index.to_numpy()
    values = series.to_numpy()
    return [[int(steps[i]), round(float(values[i]), 1)] for i in idx]


def curves(df: pd.DataFrame) -> dict[str, dict[str, list[list[float]]]]:
    return {
        env: {
            algo: _downsample(derive.curve(df, env, algo)[0], CURVE_POINTS)
            for algo in ALGOS
        }
        for env in derive.ALL_TASKS
    }


def returns(df: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    for env in derive.ALL_TASKS:
        cells = [
            [round(value) for value in derive.final_return(df, env, algo)]
            for algo in ALGOS
        ]
        rows.append([derive.short(env), cells])
    return rows


def kill(df: pd.DataFrame) -> list[list[object]]:
    return [
        [
            derive.short(env),
            round(float(derive.partition(df, env, "i_kill")[0].max()), 3),
        ]
        for env in derive.MUJOCO
    ]


def write_data(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = {"CURVES": curves(df), "RETURNS": returns(df), "KILL": kill(df)}
    lines = ["// Generated from analysis/datasets; do not edit by hand."]
    lines += [
        f"window.{name}={json.dumps(value, separators=(',', ':'))};"
        for name, value in blocks.items()
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def render_figures(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    return [
        *figures.equivalence(df, out_dir, ext="svg"),
        *figures.identity(df, out_dir, envs=derive.MUJOCO, ext="svg"),
        *figures.partition(df, out_dir, ext="svg"),
        *figures.beta(df, out_dir, ext="svg"),
    ]


def main() -> None:
    from analysis.render import sources

    df = sources.baselines()
    derive.require_complete(df, "baselines")
    print(write_data(df, WEBSITE / "js" / "data.js"))
    for path in render_figures(df, WEBSITE / "assets" / "figures"):
        print(path)


if __name__ == "__main__":
    main()
