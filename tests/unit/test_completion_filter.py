"""Unit tests for the pre-dispatch completion filter."""

from __future__ import annotations

from pathlib import Path

from klip_ppo.configs.sweep import JobSpecConfig
from klip_ppo.runtime.completion_filter import (
    CompletionKey,
    WandbCompletionIndex,
    default_key_resolver,
    partition_completed,
)
from klip_ppo.utils.wandb_completion import FinishedAtTargetSteps

MUJOCO_SNAPSHOT = Path(
    "configs/snapshots/presets/mujoco-baselines/walker2d__ppo_kl_per_sample.json"
)


class _FakeIndex:
    """In-memory completion index used to verify filter routing."""

    def __init__(self, complete: set[tuple[str, int]]) -> None:
        self._complete = complete
        self.calls: list[tuple[str, int]] = []

    def is_complete(self, *, group, seed, predicate) -> bool:  # noqa: ARG002
        self.calls.append((group, seed))
        return (group, seed) in self._complete


def _job(label: str, seed: int) -> JobSpecConfig:
    return JobSpecConfig(
        config_path=Path("configs/whatever.json"),
        seed=seed,
        label=label,
    )


def _key(group: str, seed: int) -> CompletionKey:
    return CompletionKey(
        entity="ent",
        project="proj",
        group=group,
        seed=seed,
        predicate=FinishedAtTargetSteps(target_steps=1000),
    )


def test_partition_routes_complete_jobs_to_skipped() -> None:
    jobs = [_job("a", 0), _job("a", 1), _job("b", 0)]
    index = _FakeIndex(complete={("a", 1), ("b", 0)})

    result = partition_completed(
        jobs,
        resolve_key=lambda spec: _key(spec.label, spec.seed),
        index_factory=lambda entity, project: index,  # type: ignore[arg-type,return-value]
    )

    assert [j.label for j in result.remaining] == ["a"]
    assert [(j.label, j.seed) for j in result.skipped] == [("a", 1), ("b", 0)]


def test_partition_keeps_jobs_with_no_resolver_key() -> None:
    jobs = [_job("a", 0), _job("b", 0)]
    index = _FakeIndex(complete=set())

    def resolver(spec: JobSpecConfig) -> CompletionKey | None:
        return None if spec.label == "a" else _key("b", 0)

    result = partition_completed(
        jobs,
        resolve_key=resolver,
        index_factory=lambda entity, project: index,  # type: ignore[arg-type,return-value]
    )

    assert [j.label for j in result.remaining] == ["a", "b"]
    assert result.skipped == ()
    assert index.calls == [("b", 0)]


def test_default_key_resolver_uses_effective_trainer_step_target(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "KLip-PPO")
    monkeypatch.setenv("WANDB_ENTITY", "KLip-PPO")
    monkeypatch.setenv("WANDB_MODE", "offline")
    spec = JobSpecConfig(
        config_path=MUJOCO_SNAPSHOT,
        seed=3,
        label="walker2d__ppo_kl_per_sample",
    )

    key = default_key_resolver(spec)

    assert key is not None
    assert isinstance(key.predicate, FinishedAtTargetSteps)
    assert key.predicate.target_steps == 999_424


def test_partition_uses_one_index_per_entity_project_pair() -> None:
    jobs = [_job("a", 0), _job("a", 1), _job("a", 2)]
    created: list[tuple[str | None, str]] = []

    def factory(entity, project):
        created.append((entity, project))
        return _FakeIndex(complete=set())

    partition_completed(
        jobs,
        resolve_key=lambda spec: _key("a", spec.seed),
        index_factory=factory,
    )

    assert created == [("ent", "proj")]


def test_default_index_factory_returns_real_index_type() -> None:
    from klip_ppo.runtime.completion_filter import _default_index_factory

    index = _default_index_factory("ent", "proj")

    assert isinstance(index, WandbCompletionIndex)
    assert index.entity == "ent"
    assert index.project == "proj"
