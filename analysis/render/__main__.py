"""Build every paper figure and table from the frozen datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.render import derive, figures, sources, tables

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "paper" / "figures"


def build(out: Path) -> list[Path]:
    baselines = sources.baselines()
    sweeps = sources.sweeps()
    derive.require_complete(baselines, "baselines")
    paths = [
        *figures.equivalence(baselines, out),
        *figures.identity(baselines, out),
        *figures.partition(baselines, out),
        *figures.beta(baselines, out),
        tables.final_returns(baselines, out),
    ]
    derive.require_complete(sweeps, "sweeps")
    paths.append(figures.sweep_knobs(sweeps, out))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for path in build(args.out):
        print(path)


if __name__ == "__main__":
    main()
