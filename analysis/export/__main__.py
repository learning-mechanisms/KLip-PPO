"""Pull the pinned runs and write the frozen datasets and run lock."""

from __future__ import annotations

import wandb

from analysis.datasets import BASELINES, RUNS_LOCK, SWEEPS
from analysis.export import download, normalize, runs


def main() -> None:
    api = wandb.Api()
    lock = runs.resolve(api)
    runs.write_lock(lock, RUNS_LOCK)
    print(
        f"pinned {len(lock['baselines'])} baseline and {len(lock['sweeps'])} sweep runs"
    )
    for cell in lock["missing"]:
        print(f"missing {cell['identity']} seed={cell['seed']}")
    normalize.write(
        normalize.normalize_baselines(download.histories(api, lock)), BASELINES
    )
    normalize.write(normalize.normalize_sweeps(download.summaries(api, lock)), SWEEPS)


if __name__ == "__main__":
    main()
