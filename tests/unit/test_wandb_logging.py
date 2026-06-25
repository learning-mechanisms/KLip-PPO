"""WandB integration tests using a fake in-process module."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.snapshot import GitInfo
from klip_ppo.runtime.local import _build_logger
from klip_ppo.utils.logging import WandbLogger
from klip_ppo.utils.snapshot import build_metadata
from klip_ppo.utils.wandb_utils import publish_file_artifact, publish_report_artifact


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None]] = []
        self.artifacts: list[tuple[FakeArtifact, list[str] | None]] = []
        self.finished = False

    def log(self, data: dict, step: int | None = None) -> None:
        self.logged.append((data, step))

    def log_artifact(
        self, artifact: FakeArtifact, aliases: list[str] | None = None
    ) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self) -> None:
        self.finished = True


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict | None = None) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str | None]] = []

    def add_file(self, path: str, name: str | None = None) -> None:
        self.files.append((path, name))


class FakeTable:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe


class FakeImage:
    def __init__(self, path: str) -> None:
        self.path = path


def install_fake_wandb(monkeypatch):
    run = FakeRun()
    calls: dict[str, list] = {"init": [], "define_metric": []}

    def init(**kwargs):
        calls["init"].append(kwargs)
        return run

    def define_metric(*args, **kwargs):
        calls["define_metric"].append((args, kwargs))

    fake = types.SimpleNamespace(
        init=init,
        define_metric=define_metric,
        Artifact=FakeArtifact,
        Table=FakeTable,
        Image=FakeImage,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return run, calls


def test_wandb_logger_logs_metrics_and_run_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    run, calls = install_fake_wandb(monkeypatch)
    run_dir = tmp_path / "run"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "logs").mkdir()
    for rel in (
        "snapshot.json",
        "metadata.json",
        "config.input.yaml",
        "stdout.log",
        "logs/console.log",
        "logs/events.jsonl",
        "metrics/train.parquet",
        "checkpoints/final.pt",
    ):
        (run_dir / rel).write_text("x")

    logger = WandbLogger(
        project="klip-ppo",
        run_name="smoke",
        group="cartpole",
        tags=("ppo", "smoke"),
        mode="offline",
        config={"experiment": "cartpole"},
        notes="test notes",
        run_dir=run_dir,
        artifact_aliases=("latest", "test"),
    )

    logger.log_iteration(
        {
            "time/env_step": 128,
            "train/return/mean": 10.0,
            "beta/scalar": None,
        }
    )
    logger.close()

    assert calls["init"][0]["project"] == "klip-ppo"
    assert calls["init"][0]["name"] == "smoke"
    assert calls["init"][0]["job_type"] == "train"
    assert calls["define_metric"][0][0] == ("time/env_step",)
    assert calls["define_metric"][1] == (
        ("*",),
        {"step_metric": "time/env_step"},
    )
    assert run.logged == [({"time/env_step": 128, "train/return/mean": 10.0}, None)]
    assert run.finished
    artifact, aliases = run.artifacts[0]
    assert artifact.type == "run"
    assert aliases == ["latest", "test"]
    assert {name for _, name in artifact.files} == {
        "snapshot.json",
        "metadata.json",
        "config.input.yaml",
        "stdout.log",
        "logs/console.log",
        "logs/events.jsonl",
        "metrics/train.parquet",
        "checkpoints/final.pt",
    }


def test_runtime_wandb_identity_uses_snapshot_source(
    tmp_path: Path, monkeypatch
) -> None:
    from datetime import UTC, datetime

    _, calls = install_fake_wandb(monkeypatch)
    cfg = ExperimentConfig.model_validate(
        {
            "name": "cartpole__ppo_clip",
            "algorithm": {"kind": "ppo_clip"},
            "env": {"id": "CartPole-v1"},
            "rollout": {"num_envs": 2, "n_steps": 32},
            "trainer": {"total_steps": 100},
            "logging": {
                "stdout": False,
                "parquet": False,
                "wandb": {"project": "klip-ppo", "mode": "offline"},
            },
        }
    )
    metadata = build_metadata(
        seed=2,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_git=GitInfo(commit="abc1234", branch="main", dirty=False),
    )
    logger = _build_logger(
        cfg,
        tmp_path / "2026-01-01T00-00-00Z__abc1234",
        seed=2,
        metadata=metadata,
        source_identity="cc-baselines__cartpole__ppo_clip",
    )
    logger.close()

    assert calls["init"][0]["group"] == "cc-baselines__cartpole__ppo_clip"
    assert (
        calls["init"][0]["name"]
        == "cc-baselines__cartpole__ppo_clip__seed=2__2026-01-01T00-00-00Z__abc1234"
    )
    assert calls["init"][0]["config"]["run"]["source_identity"] == (
        "cc-baselines__cartpole__ppo_clip"
    )


def test_publish_report_artifact_logs_report_and_tables(
    tmp_path: Path, monkeypatch
) -> None:
    run, calls = install_fake_wandb(monkeypatch)
    report = tmp_path / "2026-05-11" / "report.md"
    report.parent.mkdir()
    report.write_text("# report\n")
    returns = pd.DataFrame([{"env": "CartPole-v1", "algo": "ppo_clip", "mean": 1.0}])
    partitions = pd.DataFrame(
        [{"env": "CartPole-v1", "policy/partition/I_kill/fraction": 0.1}]
    )

    publish_report_artifact(
        report,
        project="klip-ppo",
        entity="team",
        run_name="report-run",
        mode="offline",
        aliases=("latest",),
        final_returns=returns,
        partition_stats=partitions,
        metadata={"runs_root": "artifacts/runs"},
    )

    assert calls["init"][0]["job_type"] == "report"
    artifact, aliases = run.artifacts[0]
    assert artifact.type == "report"
    assert aliases == ["latest"]
    assert artifact.files == [(str(report), "report.md")]
    assert set(run.logged[0][0]) == {"report/final_returns", "report/partition_stats"}
    assert run.finished


def test_publish_file_artifact_skips_image_preview_for_pdf(
    tmp_path: Path, monkeypatch
) -> None:
    run, calls = install_fake_wandb(monkeypatch)
    plot = tmp_path / "learning_curves.pdf"
    plot.write_bytes(b"%PDF-1.4\n")

    publish_file_artifact(
        plot,
        project="klip-ppo",
        entity="team",
        run_name="plot-run",
        mode="offline",
        job_type="plot",
        artifact_type="plot",
        aliases=("latest",),
        metadata={"kind": "learning_curves"},
        log_image_key="plot/learning_curves",
    )

    assert calls["init"][0]["job_type"] == "plot"
    artifact, aliases = run.artifacts[0]
    assert artifact.type == "plot"
    assert aliases == ["latest"]
    assert artifact.files == [(str(plot), "learning_curves.pdf")]
    assert run.logged == []
    assert run.finished
