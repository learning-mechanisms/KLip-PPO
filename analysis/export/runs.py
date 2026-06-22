"""Resolve and pin the runs that feed the datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from klip_ppo.utils.paths import SNAPSHOTS_DIR
from klip_ppo.utils.wandb_identity import source_wandb_identity

from analysis.datasets import schema

PROJECT = "klip-ppo/KLip-PPO"

DATASET_CATEGORIES = ("baselines", "sweeps")
KNOB_COLUMNS = [
    column
    for column in schema.SWEEPS
    if column not in ("env", "algo", "seed", "final_return")
]

CORE = [
    "time/env_step",
    "train/return/mean",
    "policy/kl/approx",
    "policy/clip/fraction",
    "policy/partition/I_in/fraction",
    "policy/partition/I_kill/fraction",
    "policy/partition/I_pass/fraction",
]
BETA = [
    "beta/per_sample/all/p01",
    "beta/per_sample/all/p10",
    "beta/per_sample/all/p50",
    "beta/per_sample/all/p90",
    "beta/per_sample/all/p99",
]


def _category(group: str) -> str:
    _, _, tail = group.partition("-")
    return tail


def manifest(root: Path = SNAPSHOTS_DIR) -> list[dict[str, Any]]:
    """Expected cells, one per snapshot preset, read from the config index."""
    index = json.loads((root / "_index.json").read_text())
    cells = []
    for group, names in index.items():
        if _category(group) not in DATASET_CATEGORIES:
            continue
        for name in names:
            path = root / "presets" / group / f"{name}.json"
            preset = json.loads(path.read_text())
            config = preset["config"]
            algorithm = config["algorithm"]
            cells.append(
                {
                    "identity": source_wandb_identity(path),
                    "suite": group.partition("-")[0],
                    "category": _category(group),
                    "env": config["env"]["id"],
                    "algo": algorithm["kind"],
                    "seeds": preset["seeds"],
                    "total_steps": config["trainer"]["total_steps"],
                    "knobs": {column: algorithm.get(column) for column in KNOB_COLUMNS},
                }
            )
    return cells


def _candidates(api: Any) -> dict[tuple[Any, Any], list[tuple[Any, Any, str]]]:
    """Finished, non-deprecated runs grouped by (wandb group, seed)."""
    index: dict[tuple[Any, Any], list[tuple[Any, Any, str]]] = {}
    for run in api.runs(PROJECT, filters={"state": "finished"}):
        group = run.group or ""
        if group.startswith("."):
            continue
        config = run.config
        total_steps = (config.get("trainer") or {}).get("total_steps")
        index.setdefault((group, config.get("seed")), []).append(
            (run.created_at, total_steps, run.id)
        )
    return index


def resolve(api: Any) -> dict[str, Any]:
    """Pin one finished run per expected cell, recording any that are absent."""
    candidates = _candidates(api)
    baselines, sweeps, missing = [], [], []
    for cell in manifest():
        for seed in cell["seeds"]:
            matches = [
                (created, run_id)
                for created, total_steps, run_id in candidates.get(
                    (cell["identity"], seed), []
                )
                if total_steps == cell["total_steps"]
            ]
            if not matches:
                missing.append({"identity": cell["identity"], "seed": seed})
                continue
            _, run_id = max(matches)
            entry = {
                "id": run_id,
                "env": cell["env"],
                "algo": cell["algo"],
                "seed": seed,
            }
            if cell["category"] == "sweeps":
                entry.update(cell["knobs"])
                sweeps.append(entry)
            else:
                baselines.append(entry)
    return {
        "project": PROJECT,
        "metrics": {"core": CORE, "beta": BETA},
        "baselines": sorted(baselines, key=lambda entry: entry["id"]),
        "sweeps": sorted(sweeps, key=lambda entry: entry["id"]),
        "missing": sorted(missing, key=lambda cell: (cell["identity"], cell["seed"])),
    }


def write_lock(lock: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def read_lock(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
