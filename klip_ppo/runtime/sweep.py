"""
Local Sweep runner: spawn one Job subprocess per slot.

GPU pinning is via ``CUDA_VISIBLE_DEVICES`` set on the child env, *not* via in-process
``torch.cuda.set_device``. This matches the architecture invariant: the Job is always
single-device and indexes its GPU as ``cuda:0`` regardless of which host GPU it
received.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tqdm.auto import tqdm  # type: ignore[import-untyped]

from klip_ppo.configs.sweep import GpuSlotConfig, JobSpecConfig, SweepConfig
from klip_ppo.runtime.completion_filter import (
    KeyResolver,
    default_key_resolver,
    partition_completed,
)
from klip_ppo.utils.ids import slugify, utc_timestamp
from klip_ppo.utils.log import configure_logging, get_logger, shutdown_logging
from klip_ppo.utils.paths import SWEEPS_DIR

# Job status values; "skipped" jobs do not fail the sweep.
JOB_STATUS_OK = "ok"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_SKIPPED = "skipped"


@dataclass
class JobResult:
    label: str
    seed: int
    exit_code: int
    started_at: str
    ended_at: str
    log_path: str
    slot_label: str
    status: str = JOB_STATUS_OK


@dataclass
class SweepResult:
    sweep_dir: Path
    results: list[JobResult]
    all_ok: bool


class SweepRunner:
    """Spawns at most ``concurrency`` Job subprocesses across ``slots``."""

    def __init__(
        self,
        sweep: SweepConfig,
        sweep_root: Path = SWEEPS_DIR,
        *,
        key_resolver: KeyResolver = default_key_resolver,
    ) -> None:
        if sweep.concurrency > len(sweep.slots):
            raise ValueError(
                "concurrency exceeds slot count; refusing to over-subscribe."
            )
        self.sweep = sweep
        self.key_resolver = key_resolver
        self.sweep_dir = sweep_root / f"{utc_timestamp()}__{slugify(sweep.name)}"
        self.log_dir = self.sweep_dir / "logs"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> SweepResult:
        configure_logging(
            plain_log_file=self.sweep_dir / "logs" / "console.log",
            json_log_file=self.sweep_dir / "logs" / "events.jsonl",
        )
        log = get_logger(__name__).bind(
            sweep=self.sweep.name,
            sweep_dir=str(self.sweep_dir),
        )
        self._write_manifest()

        results: list[JobResult] = []
        jobs: tuple[JobSpecConfig, ...] = self.sweep.jobs
        if self.sweep.skip_completed:
            partition = partition_completed(jobs, resolve_key=self.key_resolver)
            jobs = partition.remaining
            for spec in partition.skipped:
                results.append(_skipped_job_result(spec))
                log.info(
                    "sweep_job_skipped",
                    label=spec.label,
                    seed=spec.seed,
                    reason="wandb_already_complete",
                )

        free: list[GpuSlotConfig] = list(self.sweep.slots[: self.sweep.concurrency])
        queue: list[JobSpecConfig] = list(jobs)
        active: list[
            tuple[subprocess.Popen[bytes], JobSpecConfig, GpuSlotConfig, str, Path]
        ] = []

        log.info(
            "sweep_started",
            jobs=len(queue),
            concurrency=self.sweep.concurrency,
            slots=[slot.model_dump(mode="json") for slot in self.sweep.slots],
        )
        progress = tqdm(
            total=len(queue),
            desc=self.sweep.name,
            unit="job",
            file=sys.stdout,
            disable=not sys.stdout.isatty(),
        )
        try:
            while queue or active:
                while queue and free:
                    spec = queue.pop(0)
                    slot = free.pop(0)
                    started = datetime.now(UTC).isoformat()
                    log_path = (
                        self.log_dir / f"{slugify(spec.label)}__seed{spec.seed}.log"
                    )
                    proc = _spawn_job(
                        spec,
                        slot,
                        log_path,
                        skip_if_complete=self.sweep.skip_completed,
                    )
                    log.info(
                        "sweep_job_started",
                        label=spec.label,
                        seed=spec.seed,
                        slot=slot.label,
                        device=_slot_device_hint(slot),
                        log_path=str(log_path),
                    )
                    active.append((proc, spec, slot, started, log_path))

                if not active:
                    break

                time.sleep(0.5)
                still_active = []
                for proc, spec, slot, started, log_path in active:
                    rc = proc.poll()
                    if rc is None:
                        still_active.append((proc, spec, slot, started, log_path))
                        continue
                    ended = datetime.now(UTC).isoformat()
                    result = JobResult(
                        label=spec.label,
                        seed=spec.seed,
                        exit_code=rc,
                        started_at=started,
                        ended_at=ended,
                        log_path=str(log_path),
                        slot_label=slot.label,
                        status=JOB_STATUS_OK if rc == 0 else JOB_STATUS_FAILED,
                    )
                    results.append(result)
                    progress.update(1)
                    log_method = log.info if rc == 0 else log.error
                    log_method(
                        "sweep_job_finished",
                        label=spec.label,
                        seed=spec.seed,
                        slot=slot.label,
                        device=_slot_device_hint(slot),
                        exit_code=rc,
                        log_path=str(log_path),
                    )
                    free.append(slot)
                active = still_active

            all_ok = all(r.status != JOB_STATUS_FAILED for r in results)
            self._write_results(results)
            log.info(
                "sweep_finished",
                ok=all_ok,
                completed=sum(1 for r in results if r.status == JOB_STATUS_OK),
                skipped=sum(1 for r in results if r.status == JOB_STATUS_SKIPPED),
                failed=sum(1 for r in results if r.status == JOB_STATUS_FAILED),
            )
            return SweepResult(sweep_dir=self.sweep_dir, results=results, all_ok=all_ok)
        finally:
            progress.close()
            shutdown_logging()

    def _write_manifest(self) -> None:
        manifest = {
            "name": self.sweep.name,
            "created_at": utc_timestamp(),
            "concurrency": self.sweep.concurrency,
            "seeds": list(self.sweep.seeds),
            "slots": [s.model_dump(mode="json") for s in self.sweep.slots],
            "jobs": [s.model_dump(mode="json") for s in self.sweep.jobs],
        }
        (self.sweep_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    def _write_results(self, results: list[JobResult]) -> None:
        payload = [
            {
                "label": r.label,
                "seed": r.seed,
                "exit_code": r.exit_code,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "log_path": r.log_path,
                "slot": r.slot_label,
                "status": r.status,
            }
            for r in results
        ]
        (self.sweep_dir / "results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


def _skipped_job_result(spec: JobSpecConfig) -> JobResult:
    now = datetime.now(UTC).isoformat()
    return JobResult(
        label=spec.label,
        seed=spec.seed,
        exit_code=0,
        started_at=now,
        ended_at=now,
        log_path="",
        slot_label="",
        status=JOB_STATUS_SKIPPED,
    )


def _slot_device_hint(slot: GpuSlotConfig) -> str:
    """
    Best-effort prediction of the device the child Job will resolve to.

    Mirrors ``pick_device("auto")`` under the ``CUDA_VISIBLE_DEVICES`` we set in
    ``_spawn_job``. The child can still override via ``runtime.device`` in its config;
    ``metadata.host.effective_device`` in the run dir is the source of truth. This field
    exists so the sweep console makes the common case (auto) legible without tailing
    every child log.
    """
    import torch

    if slot.gpu_index is not None and torch.cuda.is_available():
        return f"cuda:{slot.gpu_index}"
    mps_backend = getattr(torch.backends, "mps", None)
    mps_is_available = (
        getattr(mps_backend, "is_available", None) if mps_backend else None
    )
    if callable(mps_is_available) and mps_is_available():
        return "mps"
    return "cpu"


def _spawn_job(
    spec: JobSpecConfig,
    slot: GpuSlotConfig,
    log_path: Path,
    *,
    skip_if_complete: bool = False,
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    if slot.gpu_index is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu_index)

    args = _train_args_for_job(spec, skip_if_complete=skip_if_complete)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("wb")
    return subprocess.Popen(args, stdout=log_handle, stderr=subprocess.STDOUT, env=env)


def _train_args_for_job(
    spec: JobSpecConfig, *, skip_if_complete: bool = False
) -> list[str]:
    args: list[str] = [
        sys.executable,
        "-m",
        "klip_ppo.cli.main",
        "train",
    ]
    if _is_snapshot_path(spec.config_path):
        args.extend(["--from-snapshot", str(spec.config_path)])
    else:
        args.append(str(spec.config_path))
    args.extend(["--seed", str(spec.seed), "--name", spec.label])
    for override in spec.overrides:
        args.extend(["--set", override])
    if skip_if_complete:
        args.append("--skip-if-complete")
    return args


def _is_snapshot_path(path: Path) -> bool:
    return path.suffix.lower() == ".json"
