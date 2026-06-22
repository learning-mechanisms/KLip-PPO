"""
Aggregate registry of all Python-defined presets.

A ``PresetEntry`` is ``(group, name, factory, seeds)``. ``factory()`` returns a fully
validated ``ExperimentConfig`` (a single run, with a scalar ``seed``). ``seeds`` is the
declared seed-set the preset is meant to be run over; it lives on the envelope (not in
``ExperimentConfig``) so the "one config, one process, one run" invariant the trainer
relies on stays intact. ``klip sweep`` reads ``seeds`` to fan out N runs.

Add new entries by extending the per-module ``presets()`` dicts. To override the default
seed-set for a specific preset, add an entry to ``_SEED_OVERRIDES``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.experiments import box2d, classic_control, mujoco, soft_clipping, sweeps

DEFAULT_PRESET_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Per-(group, name) overrides for the seed-set. Empty by default — every preset
# uses ``DEFAULT_PRESET_SEEDS``. Add entries here if a specific preset needs a
# different seed budget (e.g. expensive envs with a smaller set for compute reasons).
_SEED_OVERRIDES: dict[tuple[str, str], tuple[int, ...]] = {}


@dataclass(frozen=True)
class PresetEntry:
    group: str
    name: str
    factory: Callable[[], ExperimentConfig]
    seeds: tuple[int, ...] = field(default=DEFAULT_PRESET_SEEDS)

    def build(self) -> ExperimentConfig:
        return self.factory()


def _seeds_for(group: str, name: str) -> tuple[int, ...]:
    return _SEED_OVERRIDES.get((group, name), DEFAULT_PRESET_SEEDS)


def _build_registry() -> tuple[PresetEntry, ...]:
    entries: list[PresetEntry] = []

    def _add(group: str, factories: dict[str, Callable[[], ExperimentConfig]]) -> None:
        for name, factory in factories.items():
            entries.append(
                PresetEntry(
                    group=group,
                    name=name,
                    factory=factory,
                    seeds=_seeds_for(group, name),
                )
            )

    _add(classic_control.GROUP, classic_control.presets())
    _add(mujoco.GROUP, mujoco.presets())
    _add(box2d.GROUP, box2d.presets())
    _add(soft_clipping.GROUP_CC, soft_clipping.cc_presets())
    _add(soft_clipping.GROUP_MUJOCO, soft_clipping.mujoco_presets())
    _add(sweeps.GROUP_CC, sweeps.cc_presets())
    _add(sweeps.GROUP_MUJOCO, sweeps.mujoco_presets())

    _check_unique(entries)
    return tuple(entries)


def _check_unique(entries: list[PresetEntry]) -> None:
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.group, entry.name)
        if key in seen:
            raise ValueError(f"duplicate preset (group, name): {key}")
        seen.add(key)


_REGISTRY: tuple[PresetEntry, ...] | None = None


def _registry() -> tuple[PresetEntry, ...]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def iter_presets() -> Iterator[PresetEntry]:
    yield from _registry()


def preset(group: str, name: str) -> PresetEntry:
    for entry in _registry():
        if entry.group == group and entry.name == name:
            return entry
    raise KeyError(f"no preset registered with group={group!r}, name={name!r}")


def preset_groups() -> dict[str, list[str]]:
    """Return ``{group: [name, ...]}`` for help/listing."""
    out: dict[str, list[str]] = {}
    for entry in _registry():
        out.setdefault(entry.group, []).append(entry.name)
    for names in out.values():
        names.sort()
    return out
