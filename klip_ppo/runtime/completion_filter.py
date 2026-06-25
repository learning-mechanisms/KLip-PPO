"""Pre-dispatch filter that drops sweep jobs already finished on WandB."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from klip_ppo.configs.sweep import JobSpecConfig
from klip_ppo.runtime.spec_loader import load_cfg_from_spec
from klip_ppo.utils.wandb_completion import (
    CompletionPredicate,
    FinishedAtTargetSteps,
    WandbCompletionIndex,
    effective_training_env_steps,
)
from klip_ppo.utils.wandb_identity import source_wandb_identity, wandb_group


@dataclass(frozen=True)
class CompletionKey:
    """
    Inputs the filter needs to check one job against WandB.

    Decoupled from ``JobSpecConfig`` so the filter has no dependency on how the caller
    derives the ``(entity, project, group)`` triple from a config path.
    """

    entity: str | None
    project: str
    group: str
    seed: int
    predicate: CompletionPredicate


KeyResolver = Callable[[JobSpecConfig], CompletionKey | None]
"""Map one ``JobSpecConfig`` to its completion key, or ``None`` if the job should never
be skipped (for example, when WandB logging is disabled)."""

IndexFactory = Callable[[str | None, str], WandbCompletionIndex]
"""Factory for the per-``(entity, project)`` index, injectable for tests."""


def _default_index_factory(entity: str | None, project: str) -> WandbCompletionIndex:
    return WandbCompletionIndex(entity=entity, project=project)


def default_key_resolver(spec: JobSpecConfig) -> CompletionKey | None:
    """
    Reconstruct the cfg a job would run and derive its completion key.

    Returns ``None`` when WandB logging is not configured for the job, since we have no
    way to look it up; such jobs always run.
    """
    cfg = load_cfg_from_spec(spec)
    wandb_cfg = cfg.logging.wandb
    if wandb_cfg is None:
        return None
    source_identity = source_wandb_identity(spec.config_path)
    group = wandb_group(cfg, source_identity=source_identity)
    return CompletionKey(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        group=group,
        seed=spec.seed,
        predicate=FinishedAtTargetSteps(
            target_steps=effective_training_env_steps(
                total_steps=cfg.trainer.total_steps,
                num_envs=cfg.rollout.num_envs,
                n_steps=cfg.rollout.n_steps,
            )
        ),
    )


@dataclass(frozen=True)
class PartitionedJobs:
    """Result of splitting a job list into runnable vs already-complete jobs."""

    remaining: tuple[JobSpecConfig, ...]
    skipped: tuple[JobSpecConfig, ...]


def partition_completed(
    jobs: Iterable[JobSpecConfig],
    *,
    resolve_key: KeyResolver,
    index_factory: IndexFactory = _default_index_factory,
) -> PartitionedJobs:
    """
    Split ``jobs`` into ones still to run and ones already complete on WandB.

    Errors from the WandB API propagate; callers decide whether to abort the sweep or
    surface the problem (we never silently fall back to "run it").
    """
    indices: dict[tuple[str | None, str], WandbCompletionIndex] = {}
    remaining: list[JobSpecConfig] = []
    skipped: list[JobSpecConfig] = []

    for job in jobs:
        key = resolve_key(job)
        if key is None:
            remaining.append(job)
            continue
        index = indices.get((key.entity, key.project))
        if index is None:
            index = index_factory(key.entity, key.project)
            indices[(key.entity, key.project)] = index
        if index.is_complete(group=key.group, seed=key.seed, predicate=key.predicate):
            skipped.append(job)
        else:
            remaining.append(job)

    return PartitionedJobs(remaining=tuple(remaining), skipped=tuple(skipped))
