"""``klip artifacts {ls, gc}`` — browse and garbage-collect run dirs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from klip_ppo.runtime.modal_runtime import DEFAULT_VOLUME_NAME
from klip_ppo.utils.paths import ARTIFACTS_DIR, RUNS_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("ls")
def ls(
    algo: str | None = typer.Option(None, "--algo"),
    env: str | None = typer.Option(None, "--env"),
    runs_root: Path = typer.Option(RUNS_DIR, "--runs-root"),
) -> None:
    table = Table("experiment", "algo", "env", "seed", "leaf")
    for run_dir in sorted(_iter_run_dirs(runs_root)):
        parts = run_dir.relative_to(runs_root).parts
        if len(parts) != 5:
            continue
        experiment, algo_name, env_id, seed_dir, leaf = parts
        if algo is not None and algo_name != algo:
            continue
        if env is not None and env_id != env:
            continue
        table.add_row(experiment, algo_name, env_id, seed_dir, leaf)
    rprint(table)


@app.command("gc")
def gc(
    keep_last: int = typer.Option(..., "--keep-last", min=1),
    runs_root: Path = typer.Option(RUNS_DIR, "--runs-root"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
) -> None:
    """
    Per (experiment × algo × env × seed), keep the ``--keep-last`` newest leaves.

    Defaults to a dry run; pass ``--apply`` to actually delete.
    """
    grouped: dict[Path, list[Path]] = {}
    for leaf in _iter_run_dirs(runs_root):
        grouped.setdefault(leaf.parent, []).append(leaf)
    deleted: list[Path] = []
    for parent, leaves in grouped.items():
        ordered = sorted(leaves, key=lambda p: p.name, reverse=True)
        for stale in ordered[keep_last:]:
            deleted.append(stale)
            if not dry_run:
                shutil.rmtree(stale)
    rprint(
        f"[yellow]{'would delete' if dry_run else 'deleted'} {len(deleted)} run dirs[/yellow]"
    )


@app.command("pull-modal")
def pull_modal(
    remote_path: str | None = typer.Option(
        None,
        "--remote-path",
        help="Path inside the Modal Volume, e.g. /runs, /sweeps, or /runs/<...>.",
    ),
    all_: bool = typer.Option(
        False,
        "--all",
        help="Pull the whole Modal artifact volume into local artifacts/.",
    ),
    volume: str = typer.Option(DEFAULT_VOLUME_NAME, "--volume"),
    local_dest: Path | None = typer.Option(None, "--local-dest"),
    force: bool = typer.Option(True, "--force/--no-force"),
) -> None:
    """Download Modal Volume artifacts into the local artifact tree."""
    if all_:
        remote = "/"
    elif remote_path is not None:
        remote = remote_path
    else:
        raise typer.BadParameter("pass --all or --remote-path")

    destination = local_dest or _default_modal_pull_dest(remote)
    destination.mkdir(parents=True, exist_ok=True)
    cmd = ["modal", "volume", "get", volume, remote, str(destination)]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)
    rprint(f"[green]pulled[/green] modal volume={volume} remote={remote}")


def _default_modal_pull_dest(remote_path: str) -> Path:
    remote = remote_path.strip("/")
    if not remote:
        return ARTIFACTS_DIR
    return ARTIFACTS_DIR / remote


def _iter_run_dirs(runs_root: Path):
    if not runs_root.exists():
        return
    for snapshot in runs_root.rglob("snapshot.json"):
        yield snapshot.parent
