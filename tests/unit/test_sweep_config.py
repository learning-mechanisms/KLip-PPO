"""Sweep config defaults."""

from __future__ import annotations

from pathlib import Path

from klip_ppo.configs.sweep import DEFAULT_SWEEP_SEEDS, SweepConfig


def test_seedless_sweep_jobs_expand_to_default_seed_replicas() -> None:
    sweep = SweepConfig.model_validate(
        {
            "name": "cartpole-smoke",
            "jobs": [
                {
                    "config_path": "configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json",
                    "label": "cartpole__ppo_clip",
                }
            ],
            "slots": [{"label": "cpu"}],
        }
    )

    assert sweep.seeds == DEFAULT_SWEEP_SEEDS
    assert [job.seed for job in sweep.jobs] == list(DEFAULT_SWEEP_SEEDS)
    assert {job.label for job in sweep.jobs} == {"cartpole__ppo_clip"}
    assert {job.config_path for job in sweep.jobs} == {
        Path("configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json")
    }


def test_explicit_sweep_job_seed_is_not_expanded() -> None:
    sweep = SweepConfig.model_validate(
        {
            "name": "cartpole-smoke",
            "jobs": [
                {
                    "config_path": "configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json",
                    "label": "cartpole__ppo_clip",
                    "seed": 9,
                }
            ],
            "slots": [{"label": "cpu"}],
        }
    )

    assert [job.seed for job in sweep.jobs] == [9]


def test_skip_completed_defaults_false_and_round_trips() -> None:
    sweep = SweepConfig.model_validate(
        {
            "name": "cartpole-smoke",
            "skip_completed": True,
            "jobs": [
                {
                    "config_path": "configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json",
                    "label": "cartpole__ppo_clip",
                    "seed": 0,
                }
            ],
            "slots": [{"label": "cpu"}],
        }
    )

    assert sweep.skip_completed is True

    default = SweepConfig.model_validate(
        {
            "name": "cartpole-smoke",
            "jobs": [
                {
                    "config_path": "configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json",
                    "label": "cartpole__ppo_clip",
                    "seed": 0,
                }
            ],
            "slots": [{"label": "cpu"}],
        }
    )
    assert default.skip_completed is False
