"""``klip experiment-plans`` — generate staged sweep YAMLs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.table import Table

from klip_ppo.experiments.plans import (
    DEFAULT_EXPERIMENT_PLAN_DIR,
    DEFAULT_SNAPSHOT_PRESETS_ROOT,
    write_default_experiment_plans,
    write_soft_clipping_experiment_plans,
)
from klip_ppo.utils.paths import PROJECT_ROOT


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _project_relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


app = typer.Typer(add_completion=False, no_args_is_help=True)
DEFAULT_SNAPSHOT_ROOT_OPTION = _project_relative(DEFAULT_SNAPSHOT_PRESETS_ROOT)
DEFAULT_PLAN_DIR_OPTION = _project_relative(DEFAULT_EXPERIMENT_PLAN_DIR)


@app.command("generate")
def generate(
    snapshot_root: Path = typer.Option(
        DEFAULT_SNAPSHOT_ROOT_OPTION,
        "--snapshot-root",
        file_okay=False,
        dir_okay=True,
        help="Root containing materialised preset groups.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_PLAN_DIR_OPTION,
        "--output-dir",
        "--out-dir",
        file_okay=False,
        help="Directory where staged sweep YAMLs are written.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Slot device kind: 'cuda' (uses --gpu-indices), 'mps', or 'cpu'.",
    ),
    gpu_indices: str = typer.Option(
        "0",
        "--gpu-indices",
        help=(
            "Comma-separated host GPU indices (only used when --device cuda). "
            "Example: '0,1,2,3'."
        ),
    ),
    jobs_per_device: int = typer.Option(
        1,
        "--jobs-per-device",
        min=1,
        help=(
            "How many concurrent jobs share each device. Total slots = "
            "num_devices * jobs_per_device."
        ),
    ),
    concurrency: int | None = typer.Option(
        None,
        "--concurrency",
        min=1,
        help=(
            "Concurrent in-flight jobs. Defaults to the total slot count. Must "
            "be <= number of slots."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and print the plan summary without writing YAML files.",
    ),
) -> None:
    """Generate the default staged benchmark sweep plans."""
    slots = _build_slots(
        device=device, gpu_indices_raw=gpu_indices, jobs_per_device=jobs_per_device
    )
    if concurrency is not None and concurrency > len(slots):
        raise typer.BadParameter(
            f"--concurrency={concurrency} exceeds slot count {len(slots)}; "
            "increase --jobs-per-device or --gpu-indices, or lower --concurrency."
        )
    plans = write_default_experiment_plans(
        snapshot_root=_project_path(snapshot_root),
        output_dir=_project_path(output_dir),
        slots=slots,
        concurrency=concurrency,
        dry_run=dry_run,
    )

    table = Table("plan", "presets", "seeds", "jobs", "path")
    for plan in plans:
        table.add_row(
            plan.name,
            str(plan.preset_count),
            str(plan.seed_count),
            str(plan.job_count),
            _display_path(plan.path),
        )
    rprint(table)
    rprint(
        f"slots={len(slots)} concurrency={concurrency or len(slots)} device={device}"
    )
    verb = "validated" if dry_run else "wrote"
    rprint(f"[green]{verb} {len(plans)} experiment plan(s)[/green]")


@app.command("generate-soft-clipping")
def generate_soft_clipping(
    snapshot_root: Path = typer.Option(
        DEFAULT_SNAPSHOT_ROOT_OPTION,
        "--snapshot-root",
        file_okay=False,
        dir_okay=True,
        help="Root containing materialised preset groups.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_PLAN_DIR_OPTION,
        "--output-dir",
        "--out-dir",
        file_okay=False,
        help="Directory where staged soft-clipping sweep YAMLs are written.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Slot device kind: 'cuda' (uses --gpu-indices), 'mps', or 'cpu'.",
    ),
    gpu_indices: str = typer.Option(
        "0",
        "--gpu-indices",
        help=(
            "Comma-separated host GPU indices (only used when --device cuda). "
            "Example: '0,1,2,3'."
        ),
    ),
    jobs_per_device: int = typer.Option(
        1,
        "--jobs-per-device",
        min=1,
        help=(
            "How many concurrent jobs share each device. Total slots = "
            "num_devices * jobs_per_device."
        ),
    ),
    concurrency: int | None = typer.Option(
        None,
        "--concurrency",
        min=1,
        help=(
            "Concurrent in-flight jobs. Defaults to the total slot count. Must "
            "be <= number of slots."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and print the plan summary without writing YAML files.",
    ),
) -> None:
    """Generate staged workshop soft-clipping sweep plans."""
    slots = _build_slots(
        device=device, gpu_indices_raw=gpu_indices, jobs_per_device=jobs_per_device
    )
    if concurrency is not None and concurrency > len(slots):
        raise typer.BadParameter(
            f"--concurrency={concurrency} exceeds slot count {len(slots)}; "
            "increase --jobs-per-device or --gpu-indices, or lower --concurrency."
        )
    plans = write_soft_clipping_experiment_plans(
        snapshot_root=_project_path(snapshot_root),
        output_dir=_project_path(output_dir),
        slots=slots,
        concurrency=concurrency,
        dry_run=dry_run,
    )

    table = Table("plan", "presets", "seeds", "jobs", "path")
    for plan in plans:
        table.add_row(
            plan.name,
            str(plan.preset_count),
            str(plan.seed_count),
            str(plan.job_count),
            _display_path(plan.path),
        )
    rprint(table)
    rprint(
        f"slots={len(slots)} concurrency={concurrency or len(slots)} device={device}"
    )
    verb = "validated" if dry_run else "wrote"
    rprint(f"[green]{verb} {len(plans)} soft-clipping plan(s)[/green]")


def _build_slots(
    *, device: str, gpu_indices_raw: str, jobs_per_device: int
) -> list[Mapping[str, Any]]:
    kind = device.lower()
    if kind == "cuda":
        indices = _parse_gpu_indices(gpu_indices_raw)
        slots: list[Mapping[str, Any]] = []
        for index in indices:
            for replica in range(jobs_per_device):
                label = (
                    f"gpu{index}" if jobs_per_device == 1 else f"gpu{index}-{replica}"
                )
                slots.append({"label": label, "gpu_index": index})
        return slots
    if kind in {"mps", "cpu"}:
        return [
            {"label": kind if jobs_per_device == 1 else f"{kind}-{replica}"}
            for replica in range(jobs_per_device)
        ]
    raise typer.BadParameter(
        f"--device must be one of cuda, mps, cpu (got {device!r})."
    )


def _parse_gpu_indices(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise typer.BadParameter("--gpu-indices must list at least one index.")
    indices: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise typer.BadParameter(
                f"--gpu-indices entry {part!r} is not an integer."
            ) from exc
        if value < 0:
            raise typer.BadParameter(f"--gpu-indices entry {part!r} must be >= 0.")
        indices.append(value)
    return indices


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
