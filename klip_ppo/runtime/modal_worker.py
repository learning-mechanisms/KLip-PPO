"""Pixi-side worker entrypoint for Modal training containers."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.snapshot import ExecutionInfo, GitInfo
from klip_ppo.runtime.base import RunResult
from klip_ppo.runtime.local import worker_main
from klip_ppo.utils.ids import slugify


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    """Run a Modal training payload inside the Pixi environment."""
    payload = request["payload"]
    function_name = str(request["function_name"])
    modal_gpu = str(request["modal_gpu"])

    cfg = ExperimentConfig.model_validate(payload["config"])
    source_git = GitInfo.model_validate(payload["source_git"])
    execution = ExecutionInfo.model_validate(payload["execution"]).model_copy(
        update={
            "modal_function": function_name,
            "modal_gpu": modal_gpu,
            "modal_call_id": os.environ.get("MODAL_TASK_ID")
            or os.environ.get("MODAL_FUNCTION_CALL_ID"),
        }
    )

    input_yaml_path: Path | None = None
    if payload.get("input_yaml_text"):
        input_yaml_path = Path("/tmp/klip_config.input.yaml")
        input_yaml_path.write_text(str(payload["input_yaml_text"]))

    result = worker_main(
        cfg,
        seed=int(payload["seed"]),
        input_yaml_path=input_yaml_path,
        allow_overwrite=bool(payload["allow_overwrite"]),
        execution=execution,
        source_git=source_git,
        source_identity=payload.get("source_identity"),
        skip_if_complete=bool(payload.get("skip_if_complete", False)),
    )
    if payload.get("sweep_id"):
        _write_sweep_log(
            str(payload["sweep_id"]), cfg.name, int(payload["seed"]), result
        )
    return {
        "run_dir": str(result.run_dir),
        "iterations": result.iterations,
        "env_steps": result.env_steps,
        "final_return": result.final_return,
        "exit_status": result.exit_status,
    }


def _write_sweep_log(sweep_id: str, label: str, seed: int, result: RunResult) -> None:
    log_dir = Path(os.environ["KLIP_ARTIFACTS_DIR"]) / "sweeps" / sweep_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{slugify(label)}__seed{seed}.log"
    log_path.write_text(
        json.dumps(
            {
                "label": label,
                "seed": seed,
                "run_dir": str(result.run_dir),
                "exit_status": result.exit_status,
                "ended_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> None:
    """Run a request JSON file and write the result JSON file."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("usage: modal_worker REQUEST_JSON RESULT_JSON")

    request_path = Path(args[0])
    result_path = Path(args[1])
    result = run_request(json.loads(request_path.read_text()))
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
