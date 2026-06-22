"""Pytest-wide fixtures and conventions."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_only_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """All tests run on CPU and with deterministic float math."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_default_dtype(torch.float32)


@pytest.fixture
def fixed_seed() -> int:
    return 0xC0FFEE


@pytest.fixture
def tmp_artifacts(tmp_path, monkeypatch):
    """Redirect artifacts under a tmp dir for tests that write run dirs."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("klip_ppo.utils.paths.ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr("klip_ppo.utils.paths.RUNS_DIR", artifacts / "runs")
    monkeypatch.setattr("klip_ppo.utils.paths.SWEEPS_DIR", artifacts / "sweeps")
    monkeypatch.setattr("klip_ppo.utils.paths.REPORTS_DIR", artifacts / "reports")
    yield artifacts
