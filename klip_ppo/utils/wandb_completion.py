"""
Query Weights & Biases to decide whether a (group, seed) is already done.

Used by sweep runners (pre-dispatch filter) and the worker preflight to skip seeds that
already have a finished run on WandB, so a relaunch does not redo work the previous run
completed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from klip_ppo.utils.logging import WANDB_STEP_METRIC


@dataclass(frozen=True)
class RunSnapshot:
    """Minimal view of a WandB run that completion predicates consume."""

    run_id: str
    state: str
    summary: dict[str, Any]


class CompletionPredicate(Protocol):
    """A callable that decides if a finished WandB run counts as complete."""

    def __call__(self, snapshot: RunSnapshot) -> bool: ...


@dataclass(frozen=True)
class FinishedAtTargetSteps:
    """
    Predicate: state == 'finished' and summary[step_key] >= target_steps.

    ``step_key`` defaults to the run's primary step metric so the same predicate works
    for any experiment that uses the project-wide convention.
    """

    target_steps: int
    step_key: str = WANDB_STEP_METRIC

    def __call__(self, snapshot: RunSnapshot) -> bool:
        if snapshot.state != "finished":
            return False
        recorded = snapshot.summary.get(self.step_key)
        if not isinstance(recorded, (int, float)):
            return False
        return int(recorded) >= self.target_steps


def effective_training_env_steps(
    *, total_steps: int, num_envs: int, n_steps: int
) -> int:
    """
    Return the final ``time/env_step`` emitted by the PPO trainer.

    The trainer runs an integer number of full rollouts, where each rollout collects
    ``num_envs * n_steps`` transitions. This intentionally mirrors ``PPOTrainer.run`` so
    WandB completion checks compare against the step that is actually logged, not the
    nominal requested ``trainer.total_steps`` when it falls between rollout boundaries.
    """
    env_steps_per_iter = num_envs * n_steps
    if total_steps <= 0 or env_steps_per_iter <= 0:
        raise ValueError("total_steps, num_envs, and n_steps must be positive")
    total_iters = max(1, total_steps // env_steps_per_iter)
    return total_iters * env_steps_per_iter


WandbApiFactory = Callable[[], Any]
"""Zero-arg factory returning a ``wandb.Api()`` instance (overridable in tests)."""


def _default_api_factory() -> Any:
    import wandb

    return wandb.Api()


@dataclass
class WandbCompletionIndex:
    """
    Cache of WandB runs per ``(entity, project, group)``.

    One ``api.runs(...)`` call per (entity, project, group); results are stored on the
    instance so per-seed lookups are O(matches in group).

    Errors (auth, network, missing project) propagate to the caller. We never silently
    treat "lookup failed" as "not complete", since that would silently rerun all
    completed work the next time WandB is briefly unreachable.
    """

    entity: str | None
    project: str
    api_factory: WandbApiFactory = field(default=_default_api_factory)

    _api: Any = field(default=None, init=False, repr=False)
    _by_group_seed: dict[tuple[str, int], list[RunSnapshot]] = field(
        default_factory=dict, init=False, repr=False
    )
    _loaded_groups: set[str] = field(default_factory=set, init=False, repr=False)

    def is_complete(
        self,
        *,
        group: str,
        seed: int,
        predicate: CompletionPredicate,
    ) -> bool:
        """Return True iff some run in ``(group, seed)`` satisfies ``predicate``."""
        if group not in self._loaded_groups:
            self._load_group(group)
        snapshots = self._by_group_seed.get((group, seed), ())
        return any(predicate(snap) for snap in snapshots)

    def _load_group(self, group: str) -> None:
        if self._api is None:
            self._api = self.api_factory()
        path = f"{self.entity}/{self.project}" if self.entity else self.project
        runs: Iterable[Any] = self._api.runs(path, filters={"group": group})
        for run in runs:
            seed = _extract_seed(getattr(run, "config", None))
            if seed is None:
                continue
            snapshot = RunSnapshot(
                run_id=str(getattr(run, "id", "")),
                state=str(getattr(run, "state", "")),
                summary=_summary_as_dict(getattr(run, "summary", None)),
            )
            self._by_group_seed.setdefault((group, seed), []).append(snapshot)
        self._loaded_groups.add(group)


def _extract_seed(config: Any) -> int | None:
    """Pull ``config.seed`` from a WandB run config (mapping or attr-bag)."""
    if config is None:
        return None
    raw: Any
    if hasattr(config, "get"):
        raw = config.get("seed")
    else:
        raw = getattr(config, "seed", None)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    return None


def _summary_as_dict(summary: Any) -> dict[str, Any]:
    """
    Normalise a WandB run summary into a plain dict.

    The public API returns a ``SummarySubDict``-like object; convert defensively so
    predicates can rely on ``.get(...)``.
    """
    if summary is None:
        return {}
    if isinstance(summary, dict):
        return dict(summary)
    if hasattr(summary, "items"):
        return {str(k): v for k, v in summary.items()}
    return {}
