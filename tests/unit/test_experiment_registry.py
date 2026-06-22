"""
Tests for the Python preset registry.

These guard the methodological claim of the suite: that any two algorithm variants on
the same env share *exactly* the same env / network / rollout / trainer settings and the
same shared PPO knobs. The only fields allowed to differ between variants on a given env
are the algorithm-specific knobs (``clip_epsilon`` vs. ``beta`` vs. ``beta_init`` /
``kl_target``).
"""

from __future__ import annotations

from collections import defaultdict

import pytest
from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.runtime import DEFAULT_MODAL_TIMEOUT_SECONDS
from klip_ppo.experiments.registry import iter_presets, preset_groups

# Algorithm-config fields that are *expected* to differ across variants. Anything
# outside this set must match across variants on the same env.
_ALGO_VARIANT_FIELDS: frozenset[str] = frozenset(
    {
        "kind",
        "clip_epsilon",
        "clip_epsilon_for_diagnostics",
        "beta",
        "beta_init",
        "kl_target",
        "kl_penalty",
        "method",
        "softness",
        "beta_inc_factor",
        "beta_min",
        "beta_max",
        "kl_low_ratio",
        "kl_high_ratio",
    }
)


def test_registry_is_non_empty():
    entries = list(iter_presets())
    assert len(entries) >= 28, f"expected the full headline suite, got {len(entries)}"


def test_every_preset_builds_and_validates():
    """Each factory must return a fully Pydantic-validated ``ExperimentConfig``."""
    for entry in iter_presets():
        cfg = entry.build()
        assert isinstance(cfg, ExperimentConfig)
        # Name discipline: the file we write must match cfg.name.
        assert cfg.name == entry.name, (
            f"factory name mismatch: registry={entry.name!r} cfg.name={cfg.name!r}"
        )


def test_preset_names_are_globally_unique():
    seen: set[tuple[str, str]] = set()
    for entry in iter_presets():
        key = (entry.group, entry.name)
        assert key not in seen, f"duplicate (group, name): {key}"
        seen.add(key)


@pytest.mark.parametrize(
    "group", ["cc-baselines", "mujoco-baselines", "box2d-baselines"]
)
def test_variants_on_same_env_share_everything_except_algorithm_knobs(group: str):
    """Within each baseline group, group presets by env_id and check that the four
    variants on a given env agree on env / network / rollout / trainer config and on all
    shared algorithm knobs."""
    by_env: dict[str, list[ExperimentConfig]] = defaultdict(list)
    for entry in iter_presets():
        if entry.group != group:
            continue
        cfg = entry.build()
        by_env[cfg.env.id].append(cfg)

    assert by_env, f"no presets found in group {group!r}"

    for env_id, cfgs in by_env.items():
        assert len(cfgs) >= 2, f"need >=2 variants on {env_id} to check parity"
        ref = cfgs[0]
        for other in cfgs[1:]:
            assert other.env == ref.env, f"env config drift on {env_id}"
            assert other.network == ref.network, f"network drift on {env_id}"
            assert other.rollout == ref.rollout, f"rollout drift on {env_id}"
            assert other.trainer == ref.trainer, f"trainer drift on {env_id}"
            _assert_shared_algo_knobs_match(env_id, ref, other)


def _assert_shared_algo_knobs_match(
    env_id: str, ref: ExperimentConfig, other: ExperimentConfig
) -> None:
    ref_fields = ref.algorithm.model_dump()
    other_fields = other.algorithm.model_dump()
    shared_keys = (set(ref_fields) & set(other_fields)) - _ALGO_VARIANT_FIELDS
    for key in shared_keys:
        assert ref_fields[key] == other_fields[key], (
            f"shared algo knob {key!r} differs on {env_id} between "
            f"{ref.algorithm.kind} and {other.algorithm.kind}: "
            f"{ref_fields[key]!r} vs {other_fields[key]!r}"
        )


def test_mujoco_baselines_use_canonical_arch_and_budget():
    """
    All MuJoCo baselines (including Humanoid) use tanh [64, 64] for 1M steps.

    Matches Schulman 2017, Engstrom 2020, and Andrychowicz 2021.
    """
    saw_humanoid = False
    for entry in iter_presets():
        if entry.group != "mujoco-baselines":
            continue
        cfg = entry.build()
        assert cfg.network.hidden_sizes == (64, 64), (
            f"{entry.name}: {cfg.network.hidden_sizes}"
        )
        assert cfg.trainer.total_steps == 1_000_000, (
            f"{entry.name}: {cfg.trainer.total_steps}"
        )
        if cfg.env.id == "Humanoid-v4":
            saw_humanoid = True
    assert saw_humanoid, "no Humanoid-v4 presets in registry"


def test_no_registry_preset_uses_tiny_total_steps():
    """Smoke-test step counts must stay out of the large-scale preset registry."""
    for entry in iter_presets():
        cfg = entry.build()
        assert cfg.trainer.total_steps >= 10_000, (
            f"{entry.group}/{entry.name} has trainer.total_steps="
            f"{cfg.trainer.total_steps}"
        )


def test_registry_presets_use_large_scale_modal_timeout():
    for entry in iter_presets():
        cfg = entry.build()
        assert cfg.runtime.modal_timeout_seconds == DEFAULT_MODAL_TIMEOUT_SECONDS, (
            f"{entry.group}/{entry.name} has runtime.modal_timeout_seconds="
            f"{cfg.runtime.modal_timeout_seconds}"
        )


def test_box2d_uses_lunarlander_v3():
    """LunarLander-v2 was deprecated in Gymnasium 1.x; presets must use v3."""
    found = False
    for entry in iter_presets():
        if entry.group == "box2d-baselines":
            cfg = entry.build()
            assert cfg.env.id == "LunarLander-v3"
            found = True
    assert found, "no Box2D presets in registry"


def test_cc_sweep_covers_full_beta_grid():
    """Every β in the documented grid is materialised under cc-sweeps."""
    from klip_ppo.experiments.sweeps import BETA_GRID

    seen_betas: set[float] = set()
    for entry in iter_presets():
        if entry.group != "cc-sweeps" or "ppo_kl_fixed" not in entry.name:
            continue
        cfg = entry.build()
        assert cfg.algorithm.kind == "ppo_kl_fixed"
        assert hasattr(cfg.algorithm, "beta")
        seen_betas.add(cfg.algorithm.beta)
    assert seen_betas == set(BETA_GRID), (
        f"missing βs: {set(BETA_GRID) - seen_betas}; "
        f"extra: {seen_betas - set(BETA_GRID)}"
    )


def test_cc_sweep_covers_full_kl_target_grid():
    from klip_ppo.experiments.sweeps import KL_TARGET_GRID

    seen: set[float] = set()
    for entry in iter_presets():
        if entry.group != "cc-sweeps" or "ppo_kl_adaptive" not in entry.name:
            continue
        cfg = entry.build()
        assert cfg.algorithm.kind == "ppo_kl_adaptive"
        assert hasattr(cfg.algorithm, "kl_target")
        seen.add(cfg.algorithm.kl_target)
    assert seen == set(KL_TARGET_GRID)


def test_mujoco_sweeps_cover_full_grids_by_env():
    from klip_ppo.experiments.sweeps import (
        BETA_GRID,
        CLIP_EPSILON_GRID,
        KL_TARGET_GRID,
    )

    expected_envs = {"Hopper-v4", "HalfCheetah-v4"}
    betas: dict[str, set[float]] = defaultdict(set)
    kl_targets: dict[str, set[float]] = defaultdict(set)
    clip_epsilons: dict[str, set[float]] = defaultdict(set)

    for entry in iter_presets():
        if entry.group != "mujoco-sweeps":
            continue
        cfg = entry.build()
        env_id = cfg.env.id
        if cfg.algorithm.kind == "ppo_kl_fixed":
            betas[env_id].add(cfg.algorithm.beta)
        elif cfg.algorithm.kind == "ppo_kl_adaptive":
            kl_targets[env_id].add(cfg.algorithm.kl_target)
        elif cfg.algorithm.kind == "ppo_clip":
            clip_epsilons[env_id].add(cfg.algorithm.clip_epsilon)

    assert set(betas) == expected_envs
    assert set(kl_targets) == expected_envs
    assert set(clip_epsilons) == expected_envs
    for env_id in expected_envs:
        assert betas[env_id] == set(BETA_GRID)
        assert kl_targets[env_id] == set(KL_TARGET_GRID)
        assert clip_epsilons[env_id] == set(CLIP_EPSILON_GRID)


def test_preset_groups_index_lists_every_entry():
    """``preset_groups()`` must enumerate every entry in ``iter_presets()``."""
    by_group = preset_groups()
    counted = sum(len(names) for names in by_group.values())
    assert counted == len(list(iter_presets()))


def test_legacy_benchmark_modules_use_materialized_group_names():
    """Older benchmark helpers should not point at stale pre-snapshot group dirs."""
    from klip_ppo.benchmarks import box2d, classic_control, mujoco

    assert classic_control.GROUP == "cc-baselines"
    assert mujoco.GROUP == "mujoco-baselines"
    assert box2d.GROUP == "box2d-baselines"
