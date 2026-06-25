"""Plot artifact tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from klip_ppo.cli import plot as plot_cli
from klip_ppo.research.plotting import plot_beta_quantile_band, plot_learning_curves
from typer.testing import CliRunner


def test_learning_curves_write_vector_pdf(tmp_path: Path) -> None:
    out = tmp_path / "learning_curves.pdf"
    df = pd.DataFrame(
        [
            {
                "env": "CartPole-v1",
                "algo": "ppo_clip",
                "seed": 0,
                "time/env_step": step,
                "train/return/mean": value,
            }
            for step, value in ((0, 10.0), (1, 20.0), (2, 30.0))
        ]
    )

    result = plot_learning_curves(df, out)

    data = out.read_bytes()
    assert result == out
    assert data.startswith(b"%PDF")
    assert b"/Subtype /Image" not in data


def test_plot_curves_cli_defaults_to_pdf(tmp_path: Path, monkeypatch) -> None:
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(plot_cli, "REPORTS_DIR", reports_dir)
    runs_root = tmp_path / "runs"
    run_dir = (
        runs_root
        / "cc-baseline"
        / "ppo_clip"
        / "CartPole-v1"
        / "seed=0"
        / "2026-05-11T00-00-00Z__abc1234"
    )
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "snapshot.json").write_text("{}")
    pd.DataFrame(
        {
            "time/env_step": [0, 1, 2],
            "train/return/mean": [10.0, 20.0, 30.0],
        }
    ).to_parquet(run_dir / "metrics" / "train.parquet")

    result = CliRunner().invoke(plot_cli.app, ["curves", "--runs-root", str(runs_root)])

    out = reports_dir / date.today().isoformat() / "learning_curves.pdf"
    assert result.exit_code == 0, result.output
    assert out.read_bytes().startswith(b"%PDF")


def test_beta_quantile_band_writes_vector_pdf(tmp_path: Path) -> None:
    out = tmp_path / "beta_band.pdf"
    rows = []
    for algo in ("ppo_clip", "ppo_kl_per_sample"):
        for seed in (0, 1):
            for step, scale in ((0, 0.1), (1, 0.2), (2, 0.3)):
                rows.append(
                    {
                        "env": "CartPole-v1",
                        "algo": algo,
                        "seed": seed,
                        "time/env_step": step,
                        "beta/per_sample/all/p01": -3.0 * scale,
                        "beta/per_sample/all/p10": -1.0 * scale,
                        "beta/per_sample/all/p50": 0.0,
                        "beta/per_sample/all/p90": 1.0 * scale,
                        "beta/per_sample/all/p99": 3.0 * scale,
                    }
                )
    df = pd.DataFrame(rows)

    result = plot_beta_quantile_band(df, out)

    assert result == out
    data = out.read_bytes()
    assert data.startswith(b"%PDF")


def test_beta_quantile_band_rejects_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "env": "CartPole-v1",
                "algo": "ppo_clip",
                "seed": 0,
                "time/env_step": 0,
            },
        ]
    )

    with pytest.raises(ValueError, match="missing quantile columns"):
        plot_beta_quantile_band(df, tmp_path / "out.pdf")


def test_plot_kl_vs_clip_cli_defaults_to_pdf(tmp_path: Path, monkeypatch) -> None:
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(plot_cli, "REPORTS_DIR", reports_dir)
    run_dir = tmp_path / "runs" / "run-0"
    (run_dir / "metrics").mkdir(parents=True)
    pd.DataFrame(
        {
            "time/env_step": [0, 1, 2],
            "policy/kl/approx": [0.01, 0.02, 0.03],
            "policy/clip/fraction": [0.1, 0.2, 0.3],
            "policy/partition/I_kill/fraction": [0.0, 0.1, 0.2],
        }
    ).to_parquet(run_dir / "metrics" / "train.parquet")

    result = CliRunner().invoke(plot_cli.app, ["kl-vs-clip", "--run-dir", str(run_dir)])

    out = reports_dir / date.today().isoformat() / "run-0__kl_vs_clip.pdf"
    assert result.exit_code == 0, result.output
    assert out.read_bytes().startswith(b"%PDF")
