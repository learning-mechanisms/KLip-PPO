"""Sweep runner launch-path regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from klip_ppo.configs.sweep import JobSpecConfig, SweepConfig
from klip_ppo.runtime.modal_runtime import (
    MODAL_RUNTIME_ENV,
    REMOTE_PROJECT_ROOT,
    _cfg_from_job,
    _input_yaml_path_for_job,
)
from klip_ppo.runtime.sweep import SweepRunner, _train_args_for_job

SNAPSHOT_PATH = Path("configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json")
YAML_PATH = Path("tests/resources/presets/equivalence/cartpole.yaml")


def test_local_sweep_launches_json_jobs_from_snapshot() -> None:
    spec = JobSpecConfig(
        config_path=SNAPSHOT_PATH,
        seed=7,
        label="cartpole__ppo_clip",
        overrides=("trainer.total_steps=512",),
    )

    args = _train_args_for_job(spec)

    assert args[:4] == [sys.executable, "-m", "klip_ppo.cli.main", "train"]
    train_idx = args.index("train")
    assert args[train_idx + 1 : train_idx + 3] == [
        "--from-snapshot",
        str(SNAPSHOT_PATH),
    ]
    assert args[-2:] == ["--set", "trainer.total_steps=512"]


def test_local_sweep_launches_yaml_jobs_as_positional_config() -> None:
    spec = JobSpecConfig(config_path=YAML_PATH, seed=7, label="equivalence")

    args = _train_args_for_job(spec)

    train_idx = args.index("train")
    assert args[train_idx + 1] == str(YAML_PATH)
    assert "--from-snapshot" not in args


def test_train_args_emit_skip_if_complete_flag() -> None:
    spec = JobSpecConfig(config_path=YAML_PATH, seed=0, label="equivalence")

    args = _train_args_for_job(spec, skip_if_complete=True)

    assert "--skip-if-complete" in args


def test_train_args_omit_skip_if_complete_by_default() -> None:
    spec = JobSpecConfig(config_path=YAML_PATH, seed=0, label="equivalence")

    args = _train_args_for_job(spec)

    assert "--skip-if-complete" not in args


def test_modal_sweep_loads_json_jobs_from_snapshot() -> None:
    spec = JobSpecConfig(
        config_path=SNAPSHOT_PATH,
        seed=9,
        label="cartpole__ppo_clip__seed9",
        overrides=("trainer.total_steps=512",),
    )

    cfg = _cfg_from_job(spec, modal_gpu="L4")

    assert cfg.env.id == "CartPole-v1"
    assert cfg.seed == 9
    assert cfg.name == "cartpole__ppo_clip__seed9"
    assert cfg.trainer.total_steps == 512
    assert cfg.runtime.backend == "modal"
    assert cfg.runtime.modal_gpu == "L4"
    assert _input_yaml_path_for_job(spec) is None


def test_modal_sweep_enables_wandb_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "klip-ppo")
    monkeypatch.setenv("WANDB_ENTITY", "example")
    monkeypatch.setenv("WANDB_MODE", "offline")
    spec = JobSpecConfig(
        config_path=SNAPSHOT_PATH,
        seed=9,
        label="cartpole__ppo_clip__seed9",
    )

    cfg = _cfg_from_job(spec, modal_gpu="L4")

    assert cfg.logging.wandb is not None
    assert cfg.logging.wandb.project == "klip-ppo"
    assert cfg.logging.wandb.entity == "example"
    assert cfg.logging.wandb.mode == "offline"


def test_modal_sweep_preserves_yaml_input_path() -> None:
    spec = JobSpecConfig(config_path=YAML_PATH, seed=3, label="equivalence")

    assert _input_yaml_path_for_job(spec) == YAML_PATH


def test_modal_runtime_does_not_shadow_modal_package() -> None:
    python_paths = MODAL_RUNTIME_ENV["PYTHONPATH"].split(":")

    assert python_paths == [REMOTE_PROJECT_ROOT]
    assert "site-packages" not in MODAL_RUNTIME_ENV["PYTHONPATH"]


def test_local_sweep_skips_jobs_marked_complete_and_records_them(
    tmp_path: Path, monkeypatch
) -> None:
    from klip_ppo.runtime.completion_filter import CompletionKey
    from klip_ppo.utils.wandb_completion import FinishedAtTargetSteps

    def fake_spawn(spec, slot, log_path, *, skip_if_complete=False):  # noqa: ARG001
        log_path.write_text("child log\n")
        return subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])

    monkeypatch.setattr("klip_ppo.runtime.sweep._spawn_job", fake_spawn)
    sweep = SweepConfig.model_validate(
        {
            "name": "smoke",
            "skip_completed": True,
            "jobs": [
                {
                    "config_path": str(SNAPSHOT_PATH),
                    "seed": 0,
                    "label": "cartpole__ppo_clip",
                },
                {
                    "config_path": str(SNAPSHOT_PATH),
                    "seed": 1,
                    "label": "cartpole__ppo_clip",
                },
            ],
            "slots": [{"label": "cpu"}],
            "concurrency": 1,
        }
    )

    def fake_resolver(spec):
        return CompletionKey(
            entity=None,
            project="proj",
            group=f"grp__{spec.label}",
            seed=spec.seed,
            predicate=FinishedAtTargetSteps(target_steps=1),
        )

    completed = {("grp__cartpole__ppo_clip", 0)}

    class _StubIndex:
        def is_complete(self, *, group, seed, predicate):  # noqa: ARG002
            return (group, seed) in completed

    def stub_index_factory(entity, project):  # noqa: ARG001
        return _StubIndex()

    def patched_partition(jobs, *, resolve_key):
        from klip_ppo.runtime.completion_filter import partition_completed

        return partition_completed(
            jobs, resolve_key=resolve_key, index_factory=stub_index_factory
        )

    monkeypatch.setattr("klip_ppo.runtime.sweep.partition_completed", patched_partition)

    runner = SweepRunner(sweep, sweep_root=tmp_path, key_resolver=fake_resolver)
    result = runner.run()

    statuses = sorted((r.seed, r.status) for r in result.results)
    assert statuses == [(0, "skipped"), (1, "ok")]
    assert result.all_ok

    results_payload = json.loads((result.sweep_dir / "results.json").read_text())
    statuses_in_file = sorted((r["seed"], r["status"]) for r in results_payload)
    assert statuses_in_file == [(0, "skipped"), (1, "ok")]


def test_local_sweep_writes_parent_structured_logs(tmp_path: Path, monkeypatch) -> None:
    def fake_spawn(spec, slot, log_path, *, skip_if_complete=False):  # noqa: ARG001
        log_path.write_text("child log\n")
        return subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])

    monkeypatch.setattr("klip_ppo.runtime.sweep._spawn_job", fake_spawn)
    sweep = SweepConfig.model_validate(
        {
            "name": "smoke",
            "jobs": [
                {
                    "config_path": str(SNAPSHOT_PATH),
                    "seed": 0,
                    "label": "cartpole__ppo_clip",
                }
            ],
            "slots": [{"label": "cpu"}],
            "concurrency": 1,
        }
    )

    result = SweepRunner(sweep, sweep_root=tmp_path).run()

    assert result.all_ok
    events_path = result.sweep_dir / "logs" / "events.jsonl"
    events = [
        json.loads(line)["event"] for line in events_path.read_text().splitlines()
    ]
    assert events == [
        "sweep_started",
        "sweep_job_started",
        "sweep_job_finished",
        "sweep_finished",
    ]
