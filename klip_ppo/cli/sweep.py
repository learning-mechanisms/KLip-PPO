"""``klip sweep`` — orchestrate many Jobs across local GPU slots."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich import print as rprint

from klip_ppo.configs.runtime import ModalGpu
from klip_ppo.configs.sweep import SweepConfig
from klip_ppo.runtime.modal_runtime import ModalSweepRunner
from klip_ppo.runtime.sweep import SweepRunner


def sweep_command(
    config: Path = typer.Argument(..., exists=True, readable=True),
    runtime: str = typer.Option(
        "local", "--runtime", help="Execution backend: 'local' or 'modal'."
    ),
    modal_gpu: ModalGpu = typer.Option(
        "L4",
        "--modal-gpu",
        help="Modal GPU type. Use 'cpu' for CPU-only Modal sweeps.",
    ),
    allow_dirty_modal: bool = typer.Option(
        False,
        "--allow-dirty-modal",
        help="Allow Modal launch from a dirty git tree and record the diff.",
    ),
    skip_completed: bool = typer.Option(
        False,
        "--skip-completed",
        help=(
            "Filter out jobs whose (wandb_group, seed) already has a finished "
            "WandB run that reached trainer.total_steps. Overrides the value "
            "from the sweep YAML when set."
        ),
    ),
) -> None:
    """Run a sweep defined by a YAML or JSON ``SweepConfig`` file."""
    if runtime not in {"local", "modal"}:
        raise typer.BadParameter("runtime must be 'local' or 'modal'")
    if config.suffix.lower() == ".json":
        data = json.loads(config.read_text())
    else:
        data = yaml.safe_load(config.read_text())
    sweep = SweepConfig.model_validate(data)
    if skip_completed:
        sweep = sweep.model_copy(update={"skip_completed": True})
    runner = (
        SweepRunner(sweep)
        if runtime == "local"
        else ModalSweepRunner(
            sweep,
            modal_gpu=modal_gpu,
            allow_dirty=allow_dirty_modal,
        )
    )
    result = runner.run()
    rprint(
        f"[green]sweep done[/green] dir={result.sweep_dir} "
        f"ok={result.all_ok} jobs={len(result.results)}"
    )
    if not result.all_ok:
        raise typer.Exit(code=1)
