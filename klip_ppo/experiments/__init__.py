"""
Python source-of-truth for benchmark presets.

The benchmark suite (one preset per env × algorithm) lives here as Pydantic-validated
factories instead of as hand-edited YAMLs. That gives:

  - import-time validation (typos / renamed fields fail mypy or at import),
  - guaranteed parity across the four PPO variants on a given env (the same
    ``_env_base`` and ``_shared_algo_knobs`` feed every variant),
  - trivial sweep construction (loop one knob over a list).

The materialised artifact is still a JSON snapshot under
``configs/snapshots/presets/<group>/<name>.json``; ``klip train --from-snapshot``
reads it exactly as before. Test-only YAML fixtures live under
``tests/resources/presets``.
"""

from klip_ppo.experiments.registry import (
    PresetEntry,
    iter_presets,
    preset,
    preset_groups,
)

__all__ = [
    "PresetEntry",
    "iter_presets",
    "preset",
    "preset_groups",
]
