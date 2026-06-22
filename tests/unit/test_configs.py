"""Config validation + round-trip tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from klip_ppo.configs.experiment import ExperimentConfig, apply_overrides, load_yaml
from klip_ppo.core.ppo.strategies import SoftClipStrategy, build_strategy

TEST_RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"
TEST_PRESET_PATHS = sorted((TEST_RESOURCES_DIR / "presets").rglob("*.yaml"))


@pytest.mark.parametrize("preset_path", TEST_PRESET_PATHS, ids=lambda p: p.stem)
def test_every_test_preset_yaml_validates(preset_path: Path):
    data = load_yaml(preset_path)
    cfg = ExperimentConfig.model_validate(data)
    assert cfg.name


def test_apply_overrides_supports_dotted_keys():
    base = {"algorithm": {"clip_epsilon": 0.2}, "trainer": {"total_steps": 100}}
    out = apply_overrides(
        base, ["algorithm.clip_epsilon=0.1", "trainer.total_steps=999"]
    )
    assert out["algorithm"]["clip_epsilon"] == 0.1
    assert out["trainer"]["total_steps"] == 999


def test_apply_overrides_parses_json_literals():
    base = {"algorithm": {"hidden": [1, 2]}}
    out = apply_overrides(base, ["algorithm.hidden=[64, 64]"])
    assert out["algorithm"]["hidden"] == [64, 64]


def test_snapshot_json_is_deterministic(tmp_path: Path):
    cfg = ExperimentConfig.model_validate(load_yaml(TEST_PRESET_PATHS[0]))
    a = cfg.to_snapshot_json()
    b = cfg.to_snapshot_json()
    assert a == b
    parsed = json.loads(a)
    keys = list(parsed.keys())
    assert keys == sorted(keys)


def test_extra_fields_are_forbidden():
    bad = {
        "name": "x",
        "algorithm": {"kind": "ppo_clip"},
        "env": {"id": "CartPole-v1"},
        "rollout": {"num_envs": 2, "n_steps": 32},
        "trainer": {"total_steps": 100},
        "this_field_does_not_exist": True,
    }
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(bad)


def test_soft_clip_algorithm_config_validates_and_dispatches():
    cfg = ExperimentConfig.model_validate(
        {
            "name": "soft",
            "algorithm": {
                "kind": "ppo_soft_clip",
                "method": "sigmoid",
                "clip_epsilon": 0.2,
                "softness": 0.05,
            },
            "env": {"id": "CartPole-v1"},
            "rollout": {"num_envs": 2, "n_steps": 32},
            "trainer": {"total_steps": 100},
        }
    )

    assert cfg.algorithm.kind == "ppo_soft_clip"
    assert cfg.algorithm.method == "sigmoid"
    assert isinstance(build_strategy(cfg.algorithm), SoftClipStrategy)


@pytest.mark.parametrize(
    "patch",
    [
        {"runtime": {"mixed_precision": "fp16"}},
        {"network": {"share_backbone": True}},
        {"trainer": {"log_every_iters": 2}},
    ],
)
def test_exposed_but_unsupported_config_fields_fail_fast(patch: dict):
    base = {
        "name": "x",
        "algorithm": {"kind": "ppo_clip"},
        "env": {"id": "CartPole-v1"},
        "rollout": {"num_envs": 2, "n_steps": 32},
        "trainer": {"total_steps": 100},
    }
    for section, values in patch.items():
        section_data = base.setdefault(section, {})
        assert isinstance(section_data, dict)
        section_data.update(values)
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(base)


def test_wandb_config_supports_run_identity_and_artifact_options():
    base = {
        "name": "x",
        "algorithm": {"kind": "ppo_clip"},
        "env": {"id": "CartPole-v1"},
        "rollout": {"num_envs": 2, "n_steps": 32},
        "trainer": {"total_steps": 100},
        "logging": {
            "wandb": {
                "project": "klip-ppo",
                "run_name": "cartpole-smoke",
                "job_type": "train",
                "upload_artifacts": True,
                "artifact_aliases": ["latest", "smoke"],
                "resume": "never",
            }
        },
    }
    cfg = ExperimentConfig.model_validate(base)
    assert cfg.logging.wandb is not None
    assert cfg.logging.wandb.project == "klip-ppo"
    assert cfg.logging.wandb.run_name == "cartpole-smoke"
    assert cfg.logging.wandb.artifact_aliases == ("latest", "smoke")


def test_extends_merges_one_level(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "name": "base",
                "algorithm": {"kind": "ppo_clip"},
                "env": {"id": "CartPole-v1"},
                "trainer": {"total_steps": 1},
            }
        )
    )
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump(
            {"_extends": "base.yaml", "name": "override", "trainer": {"total_steps": 2}}
        )
    )
    merged = load_yaml(override)
    assert merged["name"] == "override"
    assert merged["trainer"]["total_steps"] == 2
    assert merged["env"]["id"] == "CartPole-v1"
