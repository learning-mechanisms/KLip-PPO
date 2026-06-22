"""Root ``ExperimentConfig`` plus YAML loading and CLI-override helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from klip_ppo.configs.algorithm import AnyAlgorithmConfig
from klip_ppo.configs.base import BaseConfig
from klip_ppo.configs.env import EnvConfig
from klip_ppo.configs.logging_cfg import LoggingConfig
from klip_ppo.configs.network import NetworkConfig
from klip_ppo.configs.rollout import RolloutConfig
from klip_ppo.configs.runtime import RuntimeConfig
from klip_ppo.configs.trainer import TrainerConfig


class ExperimentConfig(BaseConfig):
    """Root configuration for one training Job (one seed, one process)."""

    name: str
    """Human-readable experiment identifier used in run paths."""

    seed: int = 0
    """Random seed for Python, NumPy, Torch, and environments."""

    algorithm: AnyAlgorithmConfig
    env: EnvConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    rollout: RolloutConfig
    trainer: TrainerConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tags: tuple[str, ...] = ()
    """Free-form labels carried into logging and snapshots."""

    notes: str = ""
    """Free-form experiment notes preserved in the resolved config."""

    def to_snapshot_json(self) -> str:
        """Deterministic, diff-stable JSON dump used for ``snapshot.json``."""
        payload = self.model_dump(mode="json")
        return json.dumps(payload, indent=2, sort_keys=True)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML preset, resolving a single level of ``_extends``."""
    path = Path(path).resolve()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"preset {path} is not a YAML mapping")
    extends = data.pop("_extends", None)
    if extends is None:
        return data
    base_path = (path.parent / str(extends)).resolve()
    base = load_yaml(base_path)
    return _deep_merge(base, data)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """
    Apply ``--set key.path=<json-literal>`` overrides to a config dict.

    ``key.path`` is dotted; the leaf must be addressable by walking nested dicts. The
    right-hand side is parsed with ``json.loads``, falling back to a raw string if that
    fails (so ``--set name=foo`` works without quoting).
    """
    out = _deep_copy(data)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"override {spec!r} must be of form key.path=value")
        key, raw = spec.split("=", 1)
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        _set_dotted(out, key.strip(), value)
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = _deep_copy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _deep_copy(data: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(data))


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Any = target
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            raise KeyError(f"cannot descend into {part!r}: parent is not a dict")
        cur = cur.setdefault(part, {})
    if not isinstance(cur, dict):
        raise KeyError(f"cannot set {dotted_key!r}: parent is not a dict")
    cur[parts[-1]] = value
