"""Logger protocol and concrete sinks (stdout, parquet, optional wandb)."""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from klip_ppo.core.ppo.diagnostic_metrics import BETA_QUANTILE_KEYS
from klip_ppo.utils.wandb_utils import aliases_or_none, artifact_name

WandbMode = Literal["online", "offline", "disabled", "shared"]
WANDB_STEP_METRIC = "time/env_step"

PARQUET_SCHEMA: dict[str, str] = {
    "time/env_step": "int64",
    "time/iteration": "int64",
    "time/wall_s": "float64",
    "train/return/mean": "float64",
    "train/return/raw_mean": "float64",
    "train/return/wrapped_mean": "float64",
    "train/return/iqm": "float64",
    "train/episode/len_mean": "float64",
    "train/episode/count": "int64",
    "loss/policy": "float64",
    "loss/value": "float64",
    "loss/total": "float64",
    "policy/entropy": "float64",
    "policy/kl/approx": "float64",
    "policy/kl/full_mean": "float64",
    "policy/kl/sample_mean": "float64",
    "policy/clip/fraction": "float64",
    "policy/ratio/mean": "float64",
    "policy/ratio/min": "float64",
    "policy/ratio/max": "float64",
    "policy/ratio/p05": "float64",
    "policy/ratio/p95": "float64",
    "policy/partition/I_in/fraction": "float64",
    "policy/partition/I_pass/fraction": "float64",
    "policy/partition/I_kill/fraction": "float64",
    "policy/partition/I_unclassified/fraction": "float64",
    "beta/scalar": "float64",
    "beta/abs_mean/all": "float64",
    "beta/abs_mean/I_kill": "float64",
    "beta/signed_mean/I_kill": "float64",
    **dict.fromkeys(BETA_QUANTILE_KEYS, "float64"),
    "policy/kl/penalty": "float64",
    "value/explained_variance": "float64",
    "optim/policy_grad_norm/mean": "float64",
    "optim/policy_grad_norm/std": "float64",
    "optim/policy_grad_norm/var": "float64",
    "optim/global_grad_norm/mean": "float64",
    "optim/global_grad_norm/std": "float64",
    "optim/global_grad_norm/var": "float64",
    "optim/value_grad_norm/mean": "float64",
    "soft_clip/softness": "float64",
    "soft_clip/gate/mean/all": "float64",
    "soft_clip/gate/mean/I_in": "float64",
    "soft_clip/gate/mean/I_pass": "float64",
    "soft_clip/gate/mean/I_kill": "float64",
    "soft_clip/gate/mean/I_unclassified": "float64",
    "soft_clip/effective_beta/abs_mean/all": "float64",
    "soft_clip/effective_beta/abs_mean/I_kill": "float64",
    "soft_clip/effective_beta/signed_mean/I_kill": "float64",
    "soft_clip/kl_penalty": "float64",
    "soft_clip/unclipped_branch_weight/mean/all": "float64",
    "soft_clip/unclipped_branch_weight/mean/I_in": "float64",
    "soft_clip/unclipped_branch_weight/mean/I_pass": "float64",
    "soft_clip/unclipped_branch_weight/mean/I_kill": "float64",
    "soft_clip/unclipped_branch_weight/mean/I_unclassified": "float64",
    "update/steps": "int64",
    "update/early_stopped": "float64",
    "optim/lr": "float64",
    "diagnostics/migration_rate/mean": "float64",
    "diagnostics/migration_rate/max": "float64",
    "diagnostics/policy_grad_norm_var_per_epoch/mean": "float64",
    "diagnostics/global_grad_norm_var_per_epoch/mean": "float64",
    "eval/return/mean": "float64",
    "eval/return/std": "float64",
    "eval/return/iqm": "float64",
    "eval/episode/len_mean": "float64",
    "eval/episode/count": "int64",
}

# Per-inner-epoch records emitted only under ``trainer.diagnostic_mode == "full"``.
# One row per (rollout iteration, inner epoch). ``migration_rate`` is null on
# epoch 0 (no previous epoch within the same rollout to diff against).
EPOCH_PARQUET_SCHEMA: dict[str, str] = {
    "time/iteration": "int64",
    "epoch/index": "int64",
    "epoch/samples": "int64",
    "epoch/partition/I_in/fraction": "float64",
    "epoch/partition/I_pass/fraction": "float64",
    "epoch/partition/I_kill/fraction": "float64",
    "epoch/partition/I_unclassified/fraction": "float64",
    "epoch/migration/rate": "float64",
    "epoch/migration/count": "int64",
    "epoch/optim/policy_grad_norm/mean": "float64",
    "epoch/optim/policy_grad_norm/var": "float64",
    "epoch/optim/global_grad_norm/mean": "float64",
    "epoch/optim/global_grad_norm/var": "float64",
    "epoch/policy/kl/approx_mean": "float64",
}


class Logger(Protocol):
    def log_iteration(self, row: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


@dataclass
class StdLogger:
    """Single-line per iteration to stdout, plus structured JSON to stdout.log."""

    log_file: Path | None = None
    keys: tuple[str, ...] = (
        "time/iteration",
        "time/env_step",
        "train/return/mean",
        "loss/policy",
        "loss/value",
        "policy/kl/approx",
        "policy/clip/fraction",
        "beta/scalar",
    )

    _fh: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.log_file.open("a", buffering=1)

    def log_iteration(self, row: dict[str, Any]) -> None:
        pretty = " ".join(
            f"{k}={_fmt(row[k])}" for k in self.keys if k in row and row[k] is not None
        )
        print(pretty, flush=True)
        if self._fh is not None:
            self._fh.write(json.dumps(row, default=_jsonable) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@dataclass
class ParquetLogger:
    """Buffered append-only parquet writer for per-iteration metrics."""

    path: Path
    flush_every: int = 10

    _rows: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _writer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_iteration(self, row: dict[str, Any]) -> None:
        self._rows.append({k: row.get(k) for k in PARQUET_SCHEMA})
        if len(self._rows) >= self.flush_every:
            self._flush()

    def close(self) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _flush(self) -> None:
        if not self._rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self._rows, schema=_arrow_schema())
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)
        self._rows.clear()


@dataclass
class CompositeLogger:
    """Fan-out logger that forwards each call to a list of sinks."""

    sinks: list[Logger] = field(default_factory=list)

    def log_iteration(self, row: dict[str, Any]) -> None:
        for sink in self.sinks:
            sink.log_iteration(row)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


@dataclass
class WandbLogger:
    """
    Optional Weights & Biases sink.

    Lazy-imports wandb.
    """

    project: str
    run_name: str
    entity: str | None = None
    group: str | None = None
    tags: Iterable[str] = ()
    mode: WandbMode = "online"
    config: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    job_type: str = "train"
    resume: Literal["allow", "never", "must"] = "never"
    run_dir: Path | None = None
    upload_artifacts: bool = True
    artifact_aliases: Iterable[str] = ("latest",)

    _run: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import wandb

        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            group=self.group,
            tags=list(self.tags),
            mode=self.mode,
            name=self.run_name,
            config=self.config,
            notes=self.notes,
            job_type=self.job_type,
            resume=self.resume,
            reinit=True,
        )
        wandb.define_metric(WANDB_STEP_METRIC)
        wandb.define_metric("*", step_metric=WANDB_STEP_METRIC)

    def log_iteration(self, row: dict[str, Any]) -> None:
        if self._run is None:
            return
        clean = {k: v for k, v in row.items() if v is not None}
        self._run.log(clean)

    def close(self) -> None:
        if self._run is not None:
            try:
                if (
                    self.mode != "disabled"
                    and self.upload_artifacts
                    and self.run_dir is not None
                ):
                    self._log_run_artifact()
            finally:
                self._run.finish()
            self._run = None

    def _log_run_artifact(self) -> None:
        if self._run is None or self.run_dir is None:
            return
        try:
            import wandb

            artifact = wandb.Artifact(
                name=artifact_name("run", self.run_dir.name),
                type="run",
                metadata={"run_dir": str(self.run_dir)},
            )
            added = False
            for path in _default_run_artifact_files(self.run_dir):
                if path.exists() and path.is_file():
                    artifact.add_file(
                        str(path), name=str(path.relative_to(self.run_dir))
                    )
                    added = True
            if added:
                self._run.log_artifact(
                    artifact, aliases=aliases_or_none(self.artifact_aliases)
                )
        except Exception as exc:
            warnings.warn(
                f"wandb run artifact upload failed for {self.run_dir}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


@dataclass
class EpochParquetWriter:
    """
    Append-only parquet writer for per-inner-epoch diagnostic rows.

    Used only under ``trainer.diagnostic_mode == "full"``. The trainer owns one of these
    for the duration of a run and flushes a batch of rows after each rollout update.
    Schema is defined by ``EPOCH_PARQUET_SCHEMA``.
    """

    path: Path

    _writer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = _epoch_arrow_schema()
        normalised = [{k: row.get(k) for k in EPOCH_PARQUET_SCHEMA} for row in rows]
        table = pa.Table.from_pylist(normalised, schema=schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _arrow_schema() -> Any:
    import pyarrow as pa

    type_map: dict[str, Any] = {"int64": pa.int64(), "float64": pa.float64()}
    fields = [pa.field(name, type_map[dtype]) for name, dtype in PARQUET_SCHEMA.items()]
    return pa.schema(fields)


def _epoch_arrow_schema() -> Any:
    import pyarrow as pa

    type_map: dict[str, Any] = {"int64": pa.int64(), "float64": pa.float64()}
    fields = [
        pa.field(name, type_map[dtype]) for name, dtype in EPOCH_PARQUET_SCHEMA.items()
    ]
    return pa.schema(fields)


def _default_run_artifact_files(run_dir: Path) -> tuple[Path, ...]:
    return (
        run_dir / "snapshot.json",
        run_dir / "metadata.json",
        run_dir / "config.input.yaml",
        run_dir / "stdout.log",
        run_dir / "logs" / "console.log",
        run_dir / "logs" / "events.jsonl",
        run_dir / "metrics" / "train.parquet",
        run_dir / "metrics" / "epochs.parquet",
        run_dir / "checkpoints" / "final.pt",
    )


def wall_clock() -> float:
    return time.monotonic()
