"""
Guard that the materialised JSON snapshots on disk match the Python registry.

If this test fails, someone edited ``klip_ppo/experiments/*`` (or one of the
config classes it builds) without re-running materialisation. The fix is one
command:

    pixi run materialize

This test is the static safety net behind the "Python is source of truth"
claim — without it, the on-disk snapshots can silently drift from the registry,
which is exactly the failure mode the Python-source move was supposed to prevent.

The test compares the full preset snapshot envelope. Preset snapshots are
checked in as stable inputs, so materialisation must not inject wall-clock time,
git state, lockfile hashes, host metadata, or any other run-local field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from klip_ppo.experiments.registry import iter_presets, preset
from klip_ppo.utils.paths import SNAPSHOTS_DIR
from klip_ppo.utils.snapshot import build_preset_snapshot

_PRESETS_DIR = SNAPSHOTS_DIR / "presets"
_FIX_HINT = "run `pixi run materialize` to regenerate"


def _snapshot_path(group: str, name: str) -> Path:
    return _PRESETS_DIR / group / f"{name}.json"


def _expected_snapshot(group: str, name: str) -> dict[str, object]:
    """The deterministic preset snapshot the registry currently produces."""
    entry = preset(group, name)
    return build_preset_snapshot(
        cfg=entry.build(), group=entry.group, name=entry.name, seeds=entry.seeds
    )


# Materialise the registry once at collection time so parametrize can use it.
_REGISTRY_KEYS: list[tuple[str, str]] = sorted(
    (entry.group, entry.name) for entry in iter_presets()
)


@pytest.mark.parametrize(
    "group,name", _REGISTRY_KEYS, ids=[f"{g}/{n}" for g, n in _REGISTRY_KEYS]
)
def test_snapshot_matches_registry(group: str, name: str) -> None:
    """Each registered preset must have an up-to-date JSON snapshot on disk."""
    path = _snapshot_path(group, name)
    assert path.exists(), (
        f"missing snapshot file: {path.relative_to(SNAPSHOTS_DIR.parent)} — {_FIX_HINT}"
    )

    on_disk = json.loads(path.read_text())
    expected = _expected_snapshot(group, name)

    assert on_disk == expected, (
        f"snapshot {path.relative_to(SNAPSHOTS_DIR.parent)} is out of date — "
        f"{_FIX_HINT}"
    )


def test_no_stale_snapshot_files() -> None:
    """
    No JSON under ``configs/snapshots/presets/`` may exist without a registry entry.

    Catches the reverse drift: a preset removed from Python but its on-disk JSON
    left behind, which would silently keep being trainable from snapshot.
    """
    registered = {(g, n) for g, n in _REGISTRY_KEYS}
    on_disk: set[tuple[str, str]] = set()
    if _PRESETS_DIR.exists():
        for group_dir in _PRESETS_DIR.iterdir():
            if not group_dir.is_dir():
                continue
            for snap in group_dir.glob("*.json"):
                on_disk.add((group_dir.name, snap.stem))

    stale = sorted(on_disk - registered)
    assert not stale, (
        f"stale snapshot files (no matching registry entry): {stale} — "
        "delete them, or add them back to klip_ppo/experiments/"
    )
