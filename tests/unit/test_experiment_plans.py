"""Experiment sweep-plan generation."""

from __future__ import annotations

from pathlib import Path

import yaml
from klip_ppo.configs.sweep import SweepConfig
from klip_ppo.experiments.plans import (
    SMOKE_OVERRIDES,
    build_default_experiment_plan_payloads,
    build_soft_clipping_experiment_plan_payloads,
    write_default_experiment_plans,
    write_soft_clipping_experiment_plans,
)


def test_default_experiment_plan_payloads_are_staged_and_valid(
    tmp_path: Path,
) -> None:
    snapshot_root = _snapshot_tree(tmp_path)

    plans = build_default_experiment_plan_payloads(snapshot_root=snapshot_root)

    assert [plan["name"] for plan in plans] == [
        "p00_launch_smoke_all_seed0_256steps",
        "p01_all_presets_seed0_full",
        "p10_core_headline_seeds1_4",
        "p20_tuning_sweeps_seeds1_4",
        "p30_box2d_external_validity_seeds1_4",
        "p90_everything_all5_from_scratch",
    ]
    for plan in plans:
        SweepConfig.model_validate(plan)

    by_name = {str(plan["name"]): plan for plan in plans}
    assert by_name["p00_launch_smoke_all_seed0_256steps"]["seeds"] == [0]
    assert by_name["p10_core_headline_seeds1_4"]["seeds"] == [1, 2, 3, 4]
    assert by_name["p90_everything_all5_from_scratch"]["seeds"] == [0, 1, 2, 3, 4]

    smoke_jobs = by_name["p00_launch_smoke_all_seed0_256steps"]["jobs"]
    assert len(smoke_jobs) == 6
    assert all(job["overrides"] == list(SMOKE_OVERRIDES) for job in smoke_jobs)

    assert len(by_name["p01_all_presets_seed0_full"]["jobs"]) == 6
    assert len(by_name["p10_core_headline_seeds1_4"]["jobs"]) == 12
    assert len(by_name["p20_tuning_sweeps_seeds1_4"]["jobs"]) == 8
    assert len(by_name["p30_box2d_external_validity_seeds1_4"]["jobs"]) == 4
    assert len(by_name["p90_everything_all5_from_scratch"]["jobs"]) == 30


def test_write_default_experiment_plans_writes_valid_yaml(tmp_path: Path) -> None:
    snapshot_root = _snapshot_tree(tmp_path)
    output_dir = tmp_path / "plans"

    written = write_default_experiment_plans(
        snapshot_root=snapshot_root,
        output_dir=output_dir,
        slots=[{"label": "cpu"}],
    )

    assert len(written) == 6
    first = output_dir / "p00_launch_smoke_all_seed0_256steps.yaml"
    payload = yaml.safe_load(first.read_text())
    sweep = SweepConfig.model_validate(payload)

    assert sweep.name == "p00_launch_smoke_all_seed0_256steps"
    assert sweep.slots[0].label == "cpu"
    assert sweep.slots[0].gpu_index is None
    assert sweep.concurrency == 1
    assert written[0].preset_count == 6
    assert written[0].seed_count == 1
    assert written[0].job_count == 6


def test_soft_clipping_experiment_plan_payloads_are_staged_and_valid(
    tmp_path: Path,
) -> None:
    snapshot_root = _soft_snapshot_tree(tmp_path)

    plans = build_soft_clipping_experiment_plan_payloads(snapshot_root=snapshot_root)

    assert [plan["name"] for plan in plans] == [
        "p50_workshop_soft_clip_smoke_seed0_256steps",
        "p60_workshop_soft_clip_screen_seeds0_2",
        "p61_workshop_soft_clip_diagnostics_linear_ramp_seeds0_2",
        "p62_workshop_soft_clip_confirm_precommit_linear_ramp_s0p05_all5",
    ]
    for plan in plans:
        SweepConfig.model_validate(plan)

    by_name = {str(plan["name"]): plan for plan in plans}
    assert by_name["p50_workshop_soft_clip_smoke_seed0_256steps"]["seeds"] == [0]
    assert by_name["p60_workshop_soft_clip_screen_seeds0_2"]["seeds"] == [0, 1, 2]
    assert by_name["p61_workshop_soft_clip_diagnostics_linear_ramp_seeds0_2"][
        "seeds"
    ] == [0, 1, 2]
    assert by_name["p62_workshop_soft_clip_confirm_precommit_linear_ramp_s0p05_all5"][
        "seeds"
    ] == [0, 1, 2, 3, 4]

    smoke_jobs = by_name["p50_workshop_soft_clip_smoke_seed0_256steps"]["jobs"]
    assert len(smoke_jobs) == 24
    assert all(job["overrides"] == list(SMOKE_OVERRIDES) for job in smoke_jobs)

    assert len(by_name["p60_workshop_soft_clip_screen_seeds0_2"]["jobs"]) == 126
    assert (
        len(by_name["p61_workshop_soft_clip_diagnostics_linear_ramp_seeds0_2"]["jobs"])
        == 36
    )
    assert all(
        job["overrides"] == ["trainer.diagnostic_mode=full"]
        for job in by_name["p61_workshop_soft_clip_diagnostics_linear_ramp_seeds0_2"][
            "jobs"
        ]
    )
    assert (
        len(
            by_name["p62_workshop_soft_clip_confirm_precommit_linear_ramp_s0p05_all5"][
                "jobs"
            ]
        )
        == 30
    )


def test_write_soft_clipping_experiment_plans_writes_valid_yaml(
    tmp_path: Path,
) -> None:
    snapshot_root = _soft_snapshot_tree(tmp_path)
    output_dir = tmp_path / "plans"

    written = write_soft_clipping_experiment_plans(
        snapshot_root=snapshot_root,
        output_dir=output_dir,
        slots=[{"label": "cpu"}],
    )

    assert len(written) == 4
    first = output_dir / "p50_workshop_soft_clip_smoke_seed0_256steps.yaml"
    payload = yaml.safe_load(first.read_text())
    sweep = SweepConfig.model_validate(payload)

    assert sweep.name == "p50_workshop_soft_clip_smoke_seed0_256steps"
    assert sweep.slots[0].label == "cpu"
    assert sweep.concurrency == 1
    assert written[0].preset_count == 24
    assert written[0].seed_count == 1
    assert written[0].job_count == 24


def test_multi_slot_mps_oversubscription(tmp_path: Path) -> None:
    snapshot_root = _snapshot_tree(tmp_path)

    plans = build_default_experiment_plan_payloads(
        snapshot_root=snapshot_root,
        slots=[
            {"label": "mps-0"},
            {"label": "mps-1"},
            {"label": "mps-2"},
            {"label": "mps-3"},
        ],
    )

    sweep = SweepConfig.model_validate(plans[0])
    assert len(sweep.slots) == 4
    assert sweep.concurrency == 4
    assert all(slot.gpu_index is None for slot in sweep.slots)
    assert [slot.label for slot in sweep.slots] == ["mps-0", "mps-1", "mps-2", "mps-3"]


def test_multi_gpu_oversubscription(tmp_path: Path) -> None:
    snapshot_root = _snapshot_tree(tmp_path)

    slots = [
        {"label": f"gpu{i}-{r}", "gpu_index": i} for i in (0, 1, 2, 3) for r in (0, 1)
    ]
    plans = build_default_experiment_plan_payloads(
        snapshot_root=snapshot_root, slots=slots, concurrency=8
    )

    sweep = SweepConfig.model_validate(plans[0])
    assert len(sweep.slots) == 8
    assert sweep.concurrency == 8
    assert [slot.gpu_index for slot in sweep.slots] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_concurrency_defaults_to_slot_count(tmp_path: Path) -> None:
    snapshot_root = _snapshot_tree(tmp_path)

    plans = build_default_experiment_plan_payloads(
        snapshot_root=snapshot_root,
        slots=[{"label": "a"}, {"label": "b"}, {"label": "c"}],
    )

    assert plans[0]["concurrency"] == 3


def _snapshot_tree(tmp_path: Path) -> Path:
    snapshot_root = tmp_path / "snapshots" / "presets"
    for group, names in {
        "cc-baselines": ["cartpole__ppo_clip"],
        "mujoco-baselines": ["ant__ppo_clip", "hopper__ppo_clip"],
        "box2d-baselines": ["lunarlander__ppo_clip"],
        "cc-sweeps": ["cartpole__ppo_clip__clip_eps_0p1"],
        "mujoco-sweeps": ["hopper__ppo_clip__clip_eps_0p1"],
    }.items():
        group_dir = snapshot_root / group
        group_dir.mkdir(parents=True)
        for name in names:
            (group_dir / f"{name}.json").write_text("{}\n")
    return snapshot_root


def _soft_snapshot_tree(tmp_path: Path) -> Path:
    snapshot_root = tmp_path / "snapshots" / "presets"
    for group, names in {
        "cc-baselines": ["cartpole__ppo_clip", "cartpole__ppo_kl_per_sample"],
        "mujoco-baselines": [
            "hopper__ppo_clip",
            "hopper__ppo_kl_per_sample",
            "halfcheetah__ppo_clip",
            "halfcheetah__ppo_kl_per_sample",
        ],
    }.items():
        group_dir = snapshot_root / group
        group_dir.mkdir(parents=True)
        for name in names:
            (group_dir / f"{name}.json").write_text("{}\n")

    for group, envs in {
        "cc-soft-clipping": ("cartpole",),
        "mujoco-soft-clipping": ("hopper", "halfcheetah"),
    }.items():
        group_dir = snapshot_root / group
        group_dir.mkdir(parents=True)
        for env in envs:
            for method in ("linear_ramp", "sigmoid", "soft_min"):
                for softness in ("0p01", "0p03", "0p05", "0p1"):
                    name = f"{env}__ppo_soft_clip__{method}_s{softness}"
                    (group_dir / f"{name}.json").write_text("{}\n")
    return snapshot_root
