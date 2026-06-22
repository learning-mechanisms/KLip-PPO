"""Unit tests for WandB completion predicate + index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from klip_ppo.utils.logging import WANDB_STEP_METRIC
from klip_ppo.utils.wandb_completion import (
    FinishedAtTargetSteps,
    RunSnapshot,
    WandbCompletionIndex,
    effective_training_env_steps,
)


@dataclass
class _FakeRun:
    id: str
    state: str
    config: dict[str, Any]
    summary: dict[str, Any]


class _FakeApi:
    """Stub for ``wandb.Api`` capturing query path/filters."""

    def __init__(self, runs_by_group: dict[str, list[_FakeRun]]) -> None:
        self._runs_by_group = runs_by_group
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def runs(self, path: str, filters: dict[str, Any]) -> list[_FakeRun]:
        self.calls.append((path, filters))
        group = filters["group"]
        return list(self._runs_by_group.get(group, []))


def test_finished_at_target_steps_accepts_finished_run_at_target() -> None:
    predicate = FinishedAtTargetSteps(target_steps=1000)
    snap = RunSnapshot(run_id="a", state="finished", summary={WANDB_STEP_METRIC: 1000})

    assert predicate(snap) is True


def test_finished_at_target_steps_rejects_short_run() -> None:
    predicate = FinishedAtTargetSteps(target_steps=1000)
    snap = RunSnapshot(run_id="a", state="finished", summary={WANDB_STEP_METRIC: 500})

    assert predicate(snap) is False


def test_finished_at_target_steps_rejects_crashed_run() -> None:
    predicate = FinishedAtTargetSteps(target_steps=1000)
    snap = RunSnapshot(run_id="a", state="crashed", summary={WANDB_STEP_METRIC: 2000})

    assert predicate(snap) is False


def test_finished_at_target_steps_rejects_missing_step_summary() -> None:
    predicate = FinishedAtTargetSteps(target_steps=1000)
    snap = RunSnapshot(run_id="a", state="finished", summary={})

    assert predicate(snap) is False


def test_effective_training_env_steps_matches_full_rollout_floor() -> None:
    assert (
        effective_training_env_steps(total_steps=1_000_000, num_envs=1, n_steps=2048)
        == 999_424
    )


def test_effective_training_env_steps_runs_at_least_one_rollout() -> None:
    assert effective_training_env_steps(total_steps=128, num_envs=4, n_steps=64) == 256


def test_index_returns_true_when_any_run_satisfies_predicate() -> None:
    runs = {
        "grp": [
            _FakeRun(
                id="r1",
                state="crashed",
                config={"seed": 0},
                summary={WANDB_STEP_METRIC: 1000},
            ),
            _FakeRun(
                id="r2",
                state="finished",
                config={"seed": 0},
                summary={WANDB_STEP_METRIC: 1000},
            ),
        ],
    }
    api = _FakeApi(runs)
    index = WandbCompletionIndex(entity="ent", project="proj", api_factory=lambda: api)

    assert index.is_complete(
        group="grp", seed=0, predicate=FinishedAtTargetSteps(target_steps=1000)
    )


def test_index_returns_false_for_missing_seed() -> None:
    runs = {
        "grp": [
            _FakeRun(
                id="r1",
                state="finished",
                config={"seed": 0},
                summary={WANDB_STEP_METRIC: 1000},
            ),
        ],
    }
    index = WandbCompletionIndex(
        entity="ent", project="proj", api_factory=lambda: _FakeApi(runs)
    )

    assert not index.is_complete(
        group="grp", seed=1, predicate=FinishedAtTargetSteps(target_steps=1000)
    )


def test_index_caches_one_query_per_group() -> None:
    runs = {
        "grp": [
            _FakeRun(
                id="r1",
                state="finished",
                config={"seed": 0},
                summary={WANDB_STEP_METRIC: 1000},
            ),
            _FakeRun(
                id="r2",
                state="finished",
                config={"seed": 1},
                summary={WANDB_STEP_METRIC: 1000},
            ),
        ],
    }
    api = _FakeApi(runs)
    index = WandbCompletionIndex(entity="ent", project="proj", api_factory=lambda: api)
    predicate = FinishedAtTargetSteps(target_steps=1000)

    assert index.is_complete(group="grp", seed=0, predicate=predicate)
    assert index.is_complete(group="grp", seed=1, predicate=predicate)

    assert len(api.calls) == 1
    assert api.calls[0] == ("ent/proj", {"group": "grp"})


def test_index_path_omits_entity_when_unset() -> None:
    api = _FakeApi({"grp": []})
    index = WandbCompletionIndex(entity=None, project="proj", api_factory=lambda: api)

    index.is_complete(
        group="grp", seed=0, predicate=FinishedAtTargetSteps(target_steps=1000)
    )

    assert api.calls[0][0] == "proj"
