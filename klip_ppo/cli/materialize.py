"""
``klip materialize`` — write JSON snapshots for every Python-defined preset.

The Python registry (``klip_ppo.experiments.registry``) is the source of truth for the
benchmark suite. This command walks it and writes one JSON snapshot per preset under
``configs/snapshots/presets/<group>/<name>.json``. Training reads those snapshots via
``klip train --from-snapshot``.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from klip_ppo.experiments.registry import (
    PresetEntry,
    iter_presets,
    preset_groups,
)
from klip_ppo.utils.paths import SNAPSHOTS_DIR
from klip_ppo.utils.snapshot import build_preset_snapshot, write_preset_snapshot

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("all")
def materialize_all(
    out_dir: Path = typer.Option(
        SNAPSHOTS_DIR,
        "--out-dir",
        help="Snapshots root. Each entry lands at <out_dir>/presets/<group>/<name>.json.",
    ),
    group_filter: str | None = typer.Option(
        None,
        "--group",
        help="Only materialise presets in this group (e.g. mujoco-baselines).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be written, don't write."
    ),
) -> None:
    """Materialise every registered preset (or one group) to JSON snapshots."""
    entries = [
        e for e in iter_presets() if group_filter is None or e.group == group_filter
    ]
    if not entries:
        rprint(f"[yellow]no presets matched group={group_filter!r}[/yellow]")
        raise typer.Exit(code=1)

    written: list[Path] = []
    for entry in entries:
        cfg = entry.build()
        snapshot = build_preset_snapshot(
            cfg=cfg, group=entry.group, name=entry.name, seeds=entry.seeds
        )
        path = out_dir / "presets" / entry.group / f"{entry.name}.json"
        if dry_run:
            rprint(f"[blue]would write[/blue] {path}")
            continue
        write_preset_snapshot(path, snapshot)
        written.append(path)

    if not dry_run:
        _rewrite_index(out_dir, entries)
        rprint(f"[green]materialised {len(written)} snapshot(s)[/green]")


@app.command("list")
def list_presets(
    group_filter: str | None = typer.Option(None, "--group"),
) -> None:
    """List all registered presets without writing anything."""
    table = Table("group", "name")
    for group, names in preset_groups().items():
        if group_filter is not None and group != group_filter:
            continue
        for name in names:
            table.add_row(group, name)
    rprint(table)


def _rewrite_index(out_dir: Path, entries: list[PresetEntry]) -> None:
    """
    Refresh ``<out_dir>/_index.json`` so it reflects every group with at least one
    materialised entry.

    Existing groups that this run didn't touch are preserved.
    """
    index_path = out_dir / "_index.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index: dict[str, list[str]] = json.loads(index_path.read_text())
    else:
        index = {}
    by_group: dict[str, set[str]] = {g: set(v) for g, v in index.items()}
    for entry in entries:
        by_group.setdefault(entry.group, set()).add(entry.name)
    sorted_index = {g: sorted(v) for g, v in sorted(by_group.items())}
    index_path.write_text(json.dumps(sorted_index, indent=2, sort_keys=True) + "\n")
