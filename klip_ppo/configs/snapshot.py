"""Snapshot metadata model."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from klip_ppo.configs.base import BaseConfig


class GitInfo(BaseConfig):
    commit: str
    branch: str | None
    dirty: bool
    """Whether the worktree had uncommitted changes at run start."""

    diff_truncated: str | None = None
    """Captured worktree diff when available, possibly truncated."""


class HostInfo(BaseConfig):
    hostname: str
    platform: str
    python_version: str
    torch_version: str
    cpu_count: int | None = None
    cpu_model: str | None = None
    memory_total_bytes: int | None = None
    """Total system memory in bytes when detectable."""

    cuda_available: bool
    cuda_version: str | None
    cuda_device_count: int = 0
    device_name: str | None
    """Primary accelerator device name selected for the run."""

    gpu_memory_bytes: int | None = None
    """Primary accelerator memory in bytes when detectable."""

    mps_available: bool = False
    """Whether Apple's Metal Performance Shaders backend was available."""

    effective_device: str | None = None
    """Resolved Torch device used by the training process."""


class ExecutionInfo(BaseConfig):
    backend: str = "local"
    modal_app: str | None = None
    """Modal app name for remote execution."""

    modal_function: str | None = None
    """Modal function name used for remote execution."""

    modal_volume: str | None = None
    """Modal volume that stores remote run artifacts."""

    modal_volume_mount: str | None = None
    """Mount path for the Modal artifact volume."""

    modal_gpu: str | None = None
    """Modal GPU class requested for the remote run."""

    modal_call_id: str | None = None
    """Modal call id returned by the remote invocation."""

    image_reference: str | None = Field(
        default=None,
        description="Container image reference used for remote execution.",
    )


class LockfileInfo(BaseConfig):
    pixi_lock_sha256: str | None
    """SHA-256 digest of ``pixi.lock`` captured for reproducibility."""


class SnapshotMetadata(BaseConfig):
    """
    Run-level metadata captured at start of training.

    Persisted to ``metadata.json``; complements the deterministic ``snapshot.json``
    (which only contains the resolved ``ExperimentConfig``).
    """

    seed: int
    started_at: datetime
    ended_at: datetime | None = None
    """Wall-clock timestamp when the run finished, if known."""

    wall_seconds: float | None = None
    """Elapsed runtime in seconds, if known."""

    exit_status: str = "running"
    """Lifecycle state recorded for the training run."""

    error_message: str | None = None
    """Terminal error message for failed runs."""

    last_completed_iteration: int | None = None
    """Last fully completed trainer iteration, if known."""

    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    git: GitInfo
    host: HostInfo
    lockfile: LockfileInfo
