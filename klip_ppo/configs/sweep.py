"""Sweep configuration (orchestrates many Jobs across GPU slots)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, model_validator

from klip_ppo.configs.base import BaseConfig

DEFAULT_SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


class JobSpecConfig(BaseConfig):
    """One entry in a sweep: a config file + seed + label."""

    config_path: Path
    seed: int
    label: str
    """Short identifier used in sweep logs."""

    overrides: tuple[str, ...] = ()
    """CLI-style config overrides applied before launching the job."""


class GpuSlotConfig(BaseConfig):
    """
    A slot the sweep runner can dispatch a Job into.

    ``gpu_index`` is the host-level GPU index. The Job process is launched with
    ``CUDA_VISIBLE_DEVICES=<gpu_index>`` so that inside the child only one device is
    visible (and re-indexed to ``cuda:0``). Set ``gpu_index=None`` for CPU slots.
    """

    gpu_index: int | None = None
    """Host GPU index exposed to the child process, or CPU when omitted."""

    label: str = "cpu"
    """Human-readable slot name used in scheduler logs."""


class SweepConfig(BaseConfig):
    """A list of Jobs plus a list of slots they can run on."""

    name: str
    jobs: tuple[JobSpecConfig, ...]
    slots: tuple[GpuSlotConfig, ...]
    seeds: tuple[int, ...] = DEFAULT_SWEEP_SEEDS
    """Default seeds used when a job leaves ``seed`` unspecified."""

    concurrency: Annotated[int, Field(gt=0)] = 1
    """Maximum number of jobs allowed to run at once."""

    skip_completed: bool = False
    """Skip jobs whose ``(wandb_group, seed)`` already has a complete WandB run."""

    @model_validator(mode="before")
    @classmethod
    def _expand_seedless_jobs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_jobs = data.get("jobs")
        if not isinstance(raw_jobs, (list, tuple)):
            return data

        seeds = tuple(int(seed) for seed in data.get("seeds", DEFAULT_SWEEP_SEEDS))
        expanded_jobs: list[Any] = []
        for raw_job in raw_jobs:
            if isinstance(raw_job, dict) and raw_job.get("seed") is None:
                expanded_jobs.extend({**raw_job, "seed": seed} for seed in seeds)
            else:
                expanded_jobs.append(raw_job)

        return {**data, "jobs": expanded_jobs, "seeds": seeds}

    @model_validator(mode="after")
    def _concurrency_bounded(self) -> SweepConfig:
        if self.concurrency > len(self.slots):
            raise ValueError(
                f"concurrency={self.concurrency} exceeds slot count "
                f"{len(self.slots)}; refusing to over-subscribe."
            )
        if not self.jobs:
            raise ValueError("SweepConfig.jobs must contain at least one entry.")
        if not self.slots:
            raise ValueError("SweepConfig.slots must contain at least one entry.")
        return self
