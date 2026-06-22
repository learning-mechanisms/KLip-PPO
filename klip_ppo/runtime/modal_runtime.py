"""Modal execution backend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from klip_ppo.utils.ids import slugify, utc_timestamp
from klip_ppo.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from klip_ppo.configs.experiment import ExperimentConfig
    from klip_ppo.configs.runtime import ModalGpu
    from klip_ppo.configs.snapshot import ExecutionInfo, GitInfo
    from klip_ppo.configs.sweep import JobSpecConfig, SweepConfig
    from klip_ppo.runtime.base import RunResult

APP_NAME = "klip-ppo"
ARTIFACTS_MOUNT = "/artifacts"
DEFAULT_MODAL_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_VOLUME_NAME = "klip-ppo-artifacts"
DEFAULT_WANDB_SECRET_NAME = "wandb"
MODAL_PIXI_BUILD_ENV: dict[str, str | None] = {"CONDA_OVERRIDE_CUDA": "12.0"}
REMOTE_PROJECT_ROOT = "/root/klip-ppo"
REMOTE_PIXI_ENV = f"{REMOTE_PROJECT_ROOT}/.pixi/envs/gpu"
REMOTE_PIXI_PYTHON = f"{REMOTE_PIXI_ENV}/bin/python"
MODAL_RUNTIME_ENV = {
    "KLIP_ARTIFACTS_DIR": ARTIFACTS_MOUNT,
    "PYTHONPATH": REMOTE_PROJECT_ROOT,
}

try:
    import modal
except ImportError:  # pragma: no cover - exercised only without optional dep
    modal = None  # type: ignore[assignment]


def _require_modal() -> Any:
    if modal is None:
        raise RuntimeError(
            "Modal is not installed in this environment. Install the `modal` "
            "Pixi feature or run `pixi install` after updating pixi.lock."
        )
    return modal


def _volume_name() -> str:
    return os.environ.get("KLIP_MODAL_VOLUME", DEFAULT_VOLUME_NAME)


def _wandb_secret_name() -> str | None:
    explicit = os.environ.get("KLIP_MODAL_WANDB_SECRET")
    if explicit:
        return explicit
    if os.environ.get("WANDB_PROJECT") or os.environ.get("WANDB_API_KEY"):
        return DEFAULT_WANDB_SECRET_NAME
    return None


def _modal_secrets() -> list[Any]:
    secret_name = _wandb_secret_name()
    if secret_name is None:
        return []
    modal_mod = _require_modal()
    return [modal_mod.Secret.from_name(secret_name)]


if modal is not None:
    modal_volume = modal.Volume.from_name(_volume_name(), create_if_missing=True)
    modal_secrets = _modal_secrets()
    app = modal.App(APP_NAME)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "build-essential",
            "curl",
            "git",
            "libgl1",
            "libglib2.0-0",
            "libosmesa6",
            "patchelf",
            "swig",
        )
        .run_commands(
            "curl -fsSL https://pixi.sh/install.sh | bash",
            "ln -sf /root/.pixi/bin/pixi /usr/local/bin/pixi",
            "mkdir -p /root/klip-ppo",
        )
        .add_local_file(
            PROJECT_ROOT / "pixi.toml",
            f"{REMOTE_PROJECT_ROOT}/pixi.toml",
            copy=True,
        )
        .add_local_file(
            PROJECT_ROOT / "pixi.lock",
            f"{REMOTE_PROJECT_ROOT}/pixi.lock",
            copy=True,
        )
        .add_local_file(
            PROJECT_ROOT / "pyproject.toml",
            f"{REMOTE_PROJECT_ROOT}/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            PROJECT_ROOT / "README.md",
            f"{REMOTE_PROJECT_ROOT}/README.md",
            copy=True,
        )
        .run_commands(
            f"cd {REMOTE_PROJECT_ROOT} && pixi install -e gpu --locked",
            env=MODAL_PIXI_BUILD_ENV,
        )
        .add_local_dir(
            PROJECT_ROOT / "klip_ppo",
            f"{REMOTE_PROJECT_ROOT}/klip_ppo",
            copy=True,
        )
        .add_local_dir(
            PROJECT_ROOT / "configs",
            f"{REMOTE_PROJECT_ROOT}/configs",
            copy=True,
        )
        .run_commands(
            f"cd {REMOTE_PROJECT_ROOT} && pixi run -e gpu postinstall",
            env=MODAL_PIXI_BUILD_ENV,
        )
        .env(MODAL_RUNTIME_ENV)
    )
else:  # pragma: no cover - import guard only
    app = None
    image = None
    modal_volume = None
    modal_secrets = []


@dataclass(frozen=True)
class ModalSweepResult:
    sweep_dir: Path
    results: list[RunResult]
    all_ok: bool


class ModalRuntime:
    """Adapter that launches one Job in one Modal container."""

    def __init__(self, *, allow_dirty: bool = False, show_output: bool = True) -> None:
        self.allow_dirty = allow_dirty
        self.show_output = show_output

    def run_training(
        self,
        cfg: ExperimentConfig,
        *,
        seed: int,
        input_yaml_path: Path | None = None,
        allow_overwrite: bool = False,
        execution: ExecutionInfo | None = None,
        source_git: GitInfo | None = None,
        source_identity: str | None = None,
        skip_if_complete: bool = False,
    ) -> RunResult:
        modal_mod = _require_modal()
        payload = _job_payload(
            cfg,
            seed=seed,
            input_yaml_path=input_yaml_path,
            allow_overwrite=allow_overwrite,
            allow_dirty=self.allow_dirty,
            source_git=source_git,
            execution=execution,
            source_identity=source_identity,
            skip_if_complete=skip_if_complete,
        )
        fn = _modal_train_function(cfg.runtime.modal_gpu)
        output_cm = modal_mod.enable_output() if self.show_output else nullcontext()
        with output_cm:
            with app.run():
                result = fn.remote(payload)
        return _run_result_from_payload(result)


class ModalSweepRunner:
    """Run a Sweep on Modal with one container per JobSpec."""

    def __init__(
        self,
        sweep: SweepConfig,
        *,
        modal_gpu: ModalGpu = "L4",
        allow_dirty: bool = False,
        show_output: bool = True,
    ) -> None:
        self.sweep = sweep
        self.modal_gpu = modal_gpu
        self.allow_dirty = allow_dirty
        self.show_output = show_output
        self.sweep_id = f"{utc_timestamp()}__{slugify(sweep.name)}"
        self.sweep_dir = Path(ARTIFACTS_MOUNT) / "sweeps" / self.sweep_id

    def run(self) -> ModalSweepResult:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        from klip_ppo.runtime.completion_filter import (
            default_key_resolver,
            partition_completed,
        )
        from klip_ppo.utils.log import configure_logging, get_logger, shutdown_logging
        from klip_ppo.utils.wandb_identity import source_wandb_identity

        configure_logging()
        log = get_logger(__name__).bind(
            sweep=self.sweep.name,
            runtime="modal",
            modal_gpu=self.modal_gpu,
            sweep_dir=str(self.sweep_dir),
        )
        modal_mod = _require_modal()
        self._upload_json("manifest.json", self._manifest_payload())

        skipped_results: list[RunResult] = []
        skipped_payload: list[dict[str, Any]] = []
        specs: tuple[Any, ...] = self.sweep.jobs
        if self.sweep.skip_completed:
            partition = partition_completed(specs, resolve_key=default_key_resolver)
            specs = partition.remaining
            for spec in partition.skipped:
                skipped_results.append(_skipped_run_result())
                skipped_payload.append(_skipped_result_payload(spec))
                log.info(
                    "modal_sweep_job_skipped",
                    label=spec.label,
                    seed=spec.seed,
                    reason="wandb_already_complete",
                )

        payloads = [
            _job_payload(
                _cfg_from_job(spec, modal_gpu=self.modal_gpu),
                seed=spec.seed,
                input_yaml_path=_input_yaml_path_for_job(spec),
                allow_overwrite=False,
                allow_dirty=self.allow_dirty,
                sweep_id=self.sweep_id,
                source_identity=source_wandb_identity(spec.config_path),
                skip_if_complete=self.sweep.skip_completed,
            )
            for spec in specs
        ]
        fn = _modal_train_function(self.modal_gpu)
        output_cm = modal_mod.enable_output() if self.show_output else nullcontext()
        try:
            log.info("modal_sweep_started", jobs=len(payloads))
            with output_cm:
                with app.run():
                    raw_results = list(
                        tqdm(
                            fn.map(
                                payloads,
                                order_outputs=False,
                                return_exceptions=True,
                            ),
                            total=len(payloads),
                            desc=self.sweep.name,
                            unit="job",
                            disable=not sys.stderr.isatty(),
                        )
                    )

            results_payload: list[dict[str, Any]] = list(skipped_payload)
            results: list[RunResult] = list(skipped_results)
            errors = 0
            for raw in raw_results:
                if isinstance(raw, BaseException):
                    results_payload.append(
                        {"exit_status": "error", "error_message": repr(raw)}
                    )
                    log.error("modal_sweep_job_failed", error=repr(raw))
                    errors += 1
                    continue
                result = _run_result_from_payload(raw)
                results.append(result)
                results_payload.append(raw)
                log.info(
                    "modal_sweep_job_finished",
                    run_dir=str(result.run_dir),
                    exit_status=result.exit_status,
                )
            self._upload_json("results.json", results_payload)
            failed_results = sum(
                1 for r in results if r.exit_status not in {"ok", "skipped"}
            )
            all_ok = errors == 0 and failed_results == 0
            log.info(
                "modal_sweep_finished",
                ok=all_ok,
                completed=sum(1 for r in results if r.exit_status == "ok"),
                skipped=sum(1 for r in results if r.exit_status == "skipped"),
                failed=failed_results + errors,
            )
            return ModalSweepResult(
                sweep_dir=self.sweep_dir,
                results=results,
                all_ok=all_ok,
            )
        finally:
            shutdown_logging()

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "name": self.sweep.name,
            "created_at": utc_timestamp(),
            "runtime": "modal",
            "modal_gpu": self.modal_gpu,
            "modal_volume": _volume_name(),
            "concurrency": self.sweep.concurrency,
            "seeds": list(self.sweep.seeds),
            "slots": [s.model_dump(mode="json") for s in self.sweep.slots],
            "jobs": [s.model_dump(mode="json") for s in self.sweep.jobs],
        }

    def _upload_json(self, name: str, payload: object) -> None:
        modal_mod = _require_modal()
        volume = modal_mod.Volume.from_name(_volume_name(), create_if_missing=True)
        with TemporaryDirectory() as td:
            local_path = Path(td) / name
            local_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            remote_path = f"/sweeps/{self.sweep_id}/{name}"
            with volume.batch_upload(force=True) as batch:
                batch.put_file(local_path, remote_path)


def _cfg_from_job(spec: JobSpecConfig, *, modal_gpu: ModalGpu) -> ExperimentConfig:
    from klip_ppo.runtime.spec_loader import load_cfg_from_spec

    cfg = load_cfg_from_spec(spec)
    return _with_runtime(cfg, backend="modal", modal_gpu=modal_gpu)


def _input_yaml_path_for_job(spec: JobSpecConfig) -> Path | None:
    return None if spec.config_path.suffix.lower() == ".json" else spec.config_path


def _job_payload(
    cfg: ExperimentConfig,
    *,
    seed: int,
    input_yaml_path: Path | None,
    allow_overwrite: bool,
    allow_dirty: bool,
    source_git: GitInfo | None = None,
    execution: ExecutionInfo | None = None,
    sweep_id: str | None = None,
    source_identity: str | None = None,
    skip_if_complete: bool = False,
) -> dict[str, Any]:
    from klip_ppo.utils.wandb_identity import source_wandb_identity

    git = source_git or _source_git(allow_dirty=allow_dirty)
    if source_identity is None and input_yaml_path is not None:
        source_identity = source_wandb_identity(input_yaml_path)
    return {
        "config": json.loads(cfg.to_snapshot_json()),
        "seed": seed,
        "allow_overwrite": allow_overwrite,
        "input_yaml_name": str(input_yaml_path) if input_yaml_path else None,
        "input_yaml_text": input_yaml_path.read_text() if input_yaml_path else None,
        "source_git": git.model_dump(mode="json"),
        "execution": (execution or _execution_info(cfg.runtime.modal_gpu)).model_dump(
            mode="json"
        ),
        "sweep_id": sweep_id,
        "source_identity": source_identity,
        "skip_if_complete": skip_if_complete,
    }


def _source_git(*, allow_dirty: bool) -> GitInfo:
    from klip_ppo.configs.snapshot import GitInfo
    from klip_ppo.utils.git import read_git_state

    git = read_git_state()
    if git.dirty and not allow_dirty:
        raise RuntimeError(
            "Refusing to launch Modal job from a dirty git tree. Commit/stash changes "
            "or pass --allow-dirty-modal."
        )
    return GitInfo(
        commit=git.commit,
        branch=git.branch,
        dirty=git.dirty,
        diff_truncated=git.diff_truncated if allow_dirty else None,
    )


def _execution_info(
    modal_gpu: str, *, function_name: str | None = None
) -> ExecutionInfo:
    from klip_ppo.configs.snapshot import ExecutionInfo
    from klip_ppo.utils.lockfile import pixi_lock_sha256

    return ExecutionInfo(
        backend="modal",
        modal_app=APP_NAME,
        modal_function=function_name,
        modal_volume=_volume_name(),
        modal_volume_mount=ARTIFACTS_MOUNT,
        modal_gpu=modal_gpu,
        image_reference=f"pixi-lock:{pixi_lock_sha256()}",
    )


def _run_result_from_payload(payload: dict[str, Any]) -> RunResult:
    from klip_ppo.runtime.base import RunResult

    return RunResult(
        run_dir=Path(str(payload["run_dir"])),
        iterations=int(payload["iterations"]),
        env_steps=int(payload["env_steps"]),
        final_return=payload["final_return"],
        exit_status=str(payload["exit_status"]),
    )


def _skipped_run_result() -> RunResult:
    from klip_ppo.runtime.base import RunResult

    return RunResult(
        run_dir=Path(""),
        iterations=0,
        env_steps=0,
        final_return=None,
        exit_status="skipped",
    )


def _skipped_result_payload(spec: Any) -> dict[str, Any]:
    return {
        "label": spec.label,
        "seed": spec.seed,
        "config_path": str(spec.config_path),
        "exit_status": "skipped",
        "run_dir": "",
        "iterations": 0,
        "env_steps": 0,
        "final_return": None,
    }


def _with_runtime(
    cfg: ExperimentConfig, *, backend: str, modal_gpu: ModalGpu | None = None
) -> ExperimentConfig:
    updates: dict[str, Any] = {"backend": backend}
    if modal_gpu is not None:
        updates["modal_gpu"] = modal_gpu
    runtime = cfg.runtime.model_copy(update=updates)
    return cfg.model_copy(update={"runtime": runtime})


def _modal_train_function(modal_gpu: str) -> Any:
    lookup = {
        "cpu": train_cpu,
        "T4": train_t4,
        "L4": train_l4,
        "A10": train_a10,
        "L40S": train_l40s,
        "A100": train_a100,
        "A100-40GB": train_a100_40gb,
        "A100-80GB": train_a100_80gb,
        "H100": train_h100,
    }
    try:
        return lookup[modal_gpu]
    except KeyError as exc:
        raise ValueError(f"unsupported Modal GPU: {modal_gpu}") from exc


def _run_remote_training(
    payload: dict[str, Any], *, function_name: str, modal_gpu: str
) -> dict[str, Any]:
    with TemporaryDirectory() as td:
        request_path = Path(td) / "request.json"
        result_path = Path(td) / "result.json"
        request_path.write_text(
            json.dumps(
                {
                    "function_name": function_name,
                    "modal_gpu": modal_gpu,
                    "payload": payload,
                }
            )
            + "\n"
        )
        env = os.environ.copy()
        env["KLIP_ARTIFACTS_DIR"] = ARTIFACTS_MOUNT
        env["PYTHONPATH"] = REMOTE_PROJECT_ROOT
        env["PATH"] = f"{REMOTE_PIXI_ENV}/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = _prepend_env_path(
            f"{REMOTE_PIXI_ENV}/lib", env.get("LD_LIBRARY_PATH")
        )
        completed = subprocess.run(
            [
                REMOTE_PIXI_PYTHON,
                "-m",
                "klip_ppo.runtime.modal_worker",
                str(request_path),
                str(result_path),
            ],
            cwd=REMOTE_PROJECT_ROOT,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Pixi remote worker failed with exit code {completed.returncode}"
            )
        result_payload = json.loads(result_path.read_text())

    modal_volume.commit()
    return result_payload


def _prepend_env_path(path: str, existing: str | None) -> str:
    if existing:
        return f"{path}:{existing}"
    return path


if modal is not None:

    @app.function(
        image=image,
        volumes={ARTIFACTS_MOUNT: modal_volume},
        secrets=modal_secrets,
        timeout=DEFAULT_MODAL_TIMEOUT_SECONDS,
        include_source=False,
    )
    def train_cpu(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote_training(payload, function_name="train_cpu", modal_gpu="cpu")

    @app.function(
        image=image,
        volumes={ARTIFACTS_MOUNT: modal_volume},
        secrets=modal_secrets,
        gpu="T4",
        timeout=DEFAULT_MODAL_TIMEOUT_SECONDS,
        include_source=False,
    )
    def train_t4(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote_training(payload, function_name="train_t4", modal_gpu="T4")

    @app.function(
        image=image,
        volumes={ARTIFACTS_MOUNT: modal_volume},
        secrets=modal_secrets,
        gpu="L4",
        timeout=DEFAULT_MODAL_TIMEOUT_SECONDS,
        include_source=False,
    )
    def train_l4(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote_training(payload, function_name="train_l4", modal_gpu="L4")

    @app.function(
        image=image,
        volumes={ARTIFACTS_MOUNT: modal_volume},
        secrets=modal_secrets,
        gpu="A10",
        timeout=DEFAULT_MODAL_TIMEOUT_SECONDS,
        include_source=False,
    )
    def train_a10(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote_training(payload, function_name="train_a10", modal_gpu="A10")

    @app.function(
        image=image,
        volumes={ARTIFACTS_MOUNT: modal_volume},
        secrets=modal_secrets,
        gpu="L40S",
        timeout=DEFAULT_MODAL_TIMEOUT_SECONDS,
        include_source=False,
    )
    def train_l40s(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote_training(
            payload, function_name="train_l40s", modal_gpu="L40S"
        )

    @app.function(
        image=image,
        volumes={ARTIFACTS_MOUNT: modal_volume},
        secrets=modal_secrets,
        gpu="A100",
        timeout=DEFAULT_MODAL_TIMEOUT_SECONDS,
        include_source=False,
    )
    def train_a100(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote_training(
            payload, function_name="train_a100", modal_gpu="A100"
        )

    @app.function(
        image=image,
        volumes={ARTIFACTS_MOUNT: modal_volume},
        secrets=modal_secrets,
        gpu="A100-40GB",
        timeout=DEFAULT_MODAL_TIMEOUT_SECONDS,
        include_source=False,
    )
    def train_a100_40gb(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote_training(
            payload, function_name="train_a100_40gb", modal_gpu="A100-40GB"
        )

    @app.function(
        image=image,
        volumes={ARTIFACTS_MOUNT: modal_volume},
        secrets=modal_secrets,
        gpu="A100-80GB",
        timeout=DEFAULT_MODAL_TIMEOUT_SECONDS,
        include_source=False,
    )
    def train_a100_80gb(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote_training(
            payload, function_name="train_a100_80gb", modal_gpu="A100-80GB"
        )

    @app.function(
        image=image,
        volumes={ARTIFACTS_MOUNT: modal_volume},
        secrets=modal_secrets,
        gpu="H100",
        timeout=DEFAULT_MODAL_TIMEOUT_SECONDS,
        include_source=False,
    )
    def train_h100(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote_training(
            payload, function_name="train_h100", modal_gpu="H100"
        )
else:  # pragma: no cover - type-checkable placeholders only
    train_cpu = train_t4 = train_l4 = train_a10 = train_l40s = None
    train_a100 = train_a100_40gb = train_a100_80gb = train_h100 = None
