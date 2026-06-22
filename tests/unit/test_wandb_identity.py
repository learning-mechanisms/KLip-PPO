"""WandB run identity defaults."""

from __future__ import annotations

from pathlib import Path

from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.utils.paths import SNAPSHOTS_DIR
from klip_ppo.utils.wandb_identity import (
    default_wandb_group,
    source_wandb_identity,
    wandb_group,
    wandb_run_name,
)


def test_default_wandb_identity_groups_seed_replicas() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "clip-lr3e-4",
            "algorithm": {"kind": "ppo_clip"},
            "env": {"id": "CartPole-v1"},
            "rollout": {"num_envs": 2, "n_steps": 32},
            "trainer": {"total_steps": 100},
            "logging": {"wandb": {"project": "klip-ppo"}},
        }
    )
    run_dir = Path("artifacts/runs/clip-lr3e-4/ppo_clip/CartPole-v1/seed=3/ts__abc")

    assert default_wandb_group(cfg) == "CartPole-v1__ppo_clip__clip-lr3e-4"
    assert wandb_group(cfg) == "CartPole-v1__ppo_clip__clip-lr3e-4"
    assert (
        wandb_run_name(cfg, seed=3, run_dir=run_dir)
        == "CartPole-v1__ppo_clip__clip-lr3e-4__seed=3__ts__abc"
    )
    assert (
        wandb_run_name(
            cfg,
            seed=3,
            run_dir=run_dir,
            source_identity="cc-baselines__cartpole__ppo_clip",
        )
        == "cc-baselines__cartpole__ppo_clip__seed=3__ts__abc"
    )


def test_custom_wandb_name_and_group_still_get_seeded_run_names() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "clip-lr3e-4",
            "algorithm": {"kind": "ppo_clip"},
            "env": {"id": "CartPole-v1"},
            "rollout": {"num_envs": 2, "n_steps": 32},
            "trainer": {"total_steps": 100},
            "logging": {
                "wandb": {
                    "project": "klip-ppo",
                    "group": "cartpole__ppo_clip__clip0.2",
                    "run_name": "cartpole-clip",
                }
            },
        }
    )
    run_dir = Path("artifacts/runs/clip-lr3e-4/ppo_clip/CartPole-v1/seed=4/ts__abc")

    assert wandb_group(cfg) == "cartpole__ppo_clip__clip0.2"
    assert (
        wandb_run_name(cfg, seed=4, run_dir=run_dir) == "cartpole-clip__seed=4__ts__abc"
    )


def test_source_wandb_identity_matches_snapshot_envelope() -> None:
    snapshot = SNAPSHOTS_DIR / "presets" / "cc-baselines" / "cartpole__ppo_clip.json"

    assert source_wandb_identity(snapshot) == "cc-baselines__cartpole__ppo_clip"


def test_yaml_source_wandb_identity_uses_matching_snapshot_when_present(
    tmp_path: Path,
) -> None:
    """A user-supplied YAML resolves to its matching JSON snapshot by basename."""
    fake_yaml = tmp_path / "cartpole__ppo_clip.yaml"
    fake_yaml.write_text("# stand-in YAML; the matcher only looks at the stem\n")

    assert source_wandb_identity(fake_yaml) == "cc-baselines__cartpole__ppo_clip"
