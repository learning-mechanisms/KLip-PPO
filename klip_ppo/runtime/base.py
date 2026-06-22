"""Runtime protocol shared by ``local`` and ``modal`` backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.snapshot import ExecutionInfo, GitInfo


@dataclass(frozen=True)
class RunResult:
    """Result returned by a Runtime after running one Job."""

    run_dir: Path
    iterations: int
    env_steps: int
    final_return: float | None
    exit_status: str


class Runtime(Protocol):
    """Common interface for the local + modal backends."""

    def run_training(
        self,
        cfg: ExperimentConfig,
        *,
        seed: int,
        input_yaml_path: Path | None = None,
        allow_overwrite: bool = False,
        execution: ExecutionInfo | None = None,
        source_git: GitInfo | None = None,
    ) -> RunResult: ...
