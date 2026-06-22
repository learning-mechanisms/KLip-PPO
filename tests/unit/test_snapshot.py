"""Snapshot writer determinism + round-trip tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from klip_ppo.configs.snapshot import ExecutionInfo, GitInfo
from klip_ppo.experiments.registry import preset
from klip_ppo.utils.lockfile import pixi_lock_sha256
from klip_ppo.utils.snapshot import (
    build_metadata,
    build_preset_snapshot,
    experiment_from_preset_snapshot,
    write_preset_snapshot,
)


def test_preset_snapshot_round_trip(tmp_path: Path):
    entry = preset("cc-baselines", "cartpole__ppo_clip")
    cfg = entry.build()
    snap = build_preset_snapshot(
        cfg=cfg, group=entry.group, name=entry.name, seeds=entry.seeds
    )
    snap_again = build_preset_snapshot(
        cfg=cfg, group=entry.group, name=entry.name, seeds=entry.seeds
    )
    assert snap == snap_again
    assert "git" not in snap
    assert "lockfile" not in snap
    assert "created_at" not in snap
    assert snap["seeds"] == list(entry.seeds)
    out = tmp_path / "snap.json"
    write_preset_snapshot(out, snap)
    text_a = out.read_text()
    write_preset_snapshot(out, snap)
    text_b = out.read_text()
    assert text_a == text_b
    cfg_again = experiment_from_preset_snapshot(snap)
    assert cfg_again.to_snapshot_json() == cfg.to_snapshot_json()


def test_metadata_records_execution_and_source_git():
    git = GitInfo(commit="abc1234", branch="main", dirty=False)
    execution = ExecutionInfo(
        backend="modal",
        modal_app="klip-ppo",
        modal_function="train_l4",
        modal_volume="klip-ppo-artifacts",
        modal_gpu="L4",
    )
    meta = build_metadata(
        seed=1,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        execution=execution,
        source_git=git,
    )

    assert meta.execution.backend == "modal"
    assert meta.execution.modal_gpu == "L4"
    assert meta.git.commit == "abc1234"
    assert meta.lockfile.pixi_lock_sha256 == pixi_lock_sha256()
    assert meta.host.cpu_count is not None
