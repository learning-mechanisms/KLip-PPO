"""
MuJoCo locomotion benchmarks selected in ``.prompt.ignore/report-datasets.md``.

Priority order (must-have first):     Hopper-v4, Humanoid-v4, HalfCheetah-v4,
Walker2d-v4, Ant-v4 (optional).
"""

from __future__ import annotations

from pathlib import Path

from klip_ppo.configs.sweep import DEFAULT_SWEEP_SEEDS
from klip_ppo.utils.paths import PRESETS_DIR

GROUP = "mujoco-baselines"
_PRESET_DIR = PRESETS_DIR / GROUP

ENVS = (
    "Hopper-v4",
    "Humanoid-v4",
    "HalfCheetah-v4",
    "Walker2d-v4",
    "Ant-v4",
)


def presets() -> dict[str, Path]:
    return {p.stem: p for p in sorted(_PRESET_DIR.glob("*.yaml"))}


def sweep_seeds() -> list[int]:
    return list(DEFAULT_SWEEP_SEEDS)


def sweep_algorithms() -> list[str]:
    return ["ppo_clip", "ppo_kl_fixed", "ppo_kl_adaptive", "ppo_kl_per_sample"]
