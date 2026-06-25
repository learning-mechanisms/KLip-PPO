"""
Reconstruct an ``ExperimentConfig`` from a sweep ``JobSpecConfig``.

Shared by Modal/local sweep runners and the completion-filter key resolver. The loader
is runtime-agnostic: it does not touch ``cfg.runtime.backend`` or ``modal_gpu``; runners
apply backend-specific overrides on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from klip_ppo.configs.experiment import ExperimentConfig, apply_overrides, load_yaml
from klip_ppo.configs.sweep import JobSpecConfig
from klip_ppo.utils.wandb_env import with_wandb_from_env


def load_cfg_from_spec(spec: JobSpecConfig) -> ExperimentConfig:
    """Reconstruct the ``ExperimentConfig`` a sweep job would launch with."""
    data = _config_data_from_job_path(spec.config_path)
    data = apply_overrides(data, list(spec.overrides))
    data["seed"] = spec.seed
    data["name"] = spec.label
    cfg = ExperimentConfig.model_validate(data)
    return with_wandb_from_env(cfg)


def _config_data_from_job_path(path: Path) -> dict[str, Any]:
    """Load YAML or snapshot-JSON config data from a job's ``config_path``."""
    if path.suffix.lower() != ".json":
        return load_yaml(path)

    snapshot = json.loads(path.read_text())
    if not isinstance(snapshot, dict):
        raise ValueError(f"snapshot path must contain a JSON object: {path}")
    if "config" in snapshot:
        config = snapshot["config"]
        if not isinstance(config, dict):
            raise ValueError(f"snapshot 'config' must be a JSON object: {path}")
        return dict(config)
    return dict(snapshot)
