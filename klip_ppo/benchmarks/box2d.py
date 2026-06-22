"""Box2D benchmark — LunarLander for discrete-action external validity."""

from __future__ import annotations

from pathlib import Path

from klip_ppo.configs.sweep import DEFAULT_SWEEP_SEEDS
from klip_ppo.utils.paths import PRESETS_DIR

GROUP = "box2d-baselines"
_PRESET_DIR = PRESETS_DIR / GROUP

ENVS = ("LunarLander-v3",)


def presets() -> dict[str, Path]:
    return {p.stem: p for p in sorted(_PRESET_DIR.glob("*.yaml"))}


def sweep_seeds() -> list[int]:
    return list(DEFAULT_SWEEP_SEEDS)


def sweep_algorithms() -> list[str]:
    return ["ppo_clip", "ppo_kl_fixed", "ppo_kl_adaptive", "ppo_kl_per_sample"]
