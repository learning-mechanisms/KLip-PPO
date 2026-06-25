"""``klip snapshot {freeze, list, show}`` — preset snapshot management."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from klip_ppo.cli._common import load_experiment_from_yaml
from klip_ppo.experiments.registry import DEFAULT_PRESET_SEEDS
from klip_ppo.utils.paths import SNAPSHOTS_DIR
from klip_ppo.utils.snapshot import (
    build_preset_snapshot,
    load_preset_snapshot,
    write_preset_snapshot,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("freeze")
def freeze(
    preset: Path = typer.Option(..., "--preset", exists=True, readable=True),
    group: str = typer.Option(..., "--group"),
    name: str | None = typer.Option(None, "--name"),
    overrides: list[str] = typer.Option([], "--set"),
    seeds: str = typer.Option(
        ",".join(str(s) for s in DEFAULT_PRESET_SEEDS),
        "--seeds",
        help="Comma-separated seed-set declared on the snapshot envelope.",
    ),
) -> None:
    """Freeze a preset YAML into a deterministic JSON snapshot."""
    cfg = load_experiment_from_yaml(preset, overrides=overrides)
    snapshot_name = name or preset.stem
    seed_tuple = tuple(int(s.strip()) for s in seeds.split(",") if s.strip())
    snapshot = build_preset_snapshot(
        cfg=cfg, group=group, name=snapshot_name, seeds=seed_tuple
    )
    out = SNAPSHOTS_DIR / "presets" / group / f"{snapshot_name}.json"
    write_preset_snapshot(out, snapshot)
    _update_index(group, snapshot_name)
    rprint(f"[green]froze[/green] {out}")


@app.command("list")
def list_snapshots(
    group: str | None = typer.Option(None, "--group"),
) -> None:
    root = SNAPSHOTS_DIR / "presets"
    if not root.exists():
        rprint("[yellow]no snapshots yet[/yellow]")
        return
    table = Table("group", "name", "path")
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        if group is not None and group_dir.name != group:
            continue
        for snap in sorted(group_dir.glob("*.json")):
            table.add_row(group_dir.name, snap.stem, str(snap))
    rprint(table)


@app.command("show")
def show(name: str = typer.Argument(...)) -> None:
    root = SNAPSHOTS_DIR / "presets"
    matches = list(root.rglob(f"{name}.json"))
    if not matches:
        raise typer.BadParameter(f"no snapshot named {name!r} under {root}")
    if len(matches) > 1:
        rprint(
            "[yellow]multiple matches:[/yellow] " + ", ".join(str(p) for p in matches)
        )
    snapshot = load_preset_snapshot(matches[0])
    rprint(json.dumps(snapshot, indent=2, sort_keys=True))


def _update_index(group: str, snapshot_name: str) -> None:
    index_path = SNAPSHOTS_DIR / "_index.json"
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index: dict[str, list[str]] = json.loads(index_path.read_text())
    else:
        index = {}
    entries = set(index.get(group, []))
    entries.add(snapshot_name)
    index[group] = sorted(entries)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
