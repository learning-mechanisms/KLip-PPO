"""End-to-end smoke training on CartPole-v1 (CPU, ~1 second)."""

from __future__ import annotations

import json

import pytest
from klip_ppo.configs.experiment import ExperimentConfig, apply_overrides
from klip_ppo.core.checkpoint import CheckpointManager
from klip_ppo.experiments.registry import preset as preset_entry
from klip_ppo.runtime.local import worker_main

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _cartpole_dict(kind: str) -> dict:
    """Pull a CartPole baseline config from the Python registry as a dict."""
    cfg = preset_entry("cc-baselines", f"cartpole__{kind}").build()
    return json.loads(cfg.to_snapshot_json())


def _cartpole_soft_dict(method: str) -> dict:
    """Pull a CartPole soft-clipping config from the Python registry as a dict."""
    cfg = preset_entry(
        "cc-soft-clipping", f"cartpole__ppo_soft_clip__{method}_s0p05"
    ).build()
    return json.loads(cfg.to_snapshot_json())


@pytest.mark.parametrize(
    "kind",
    ["ppo_clip", "ppo_kl_fixed", "ppo_kl_adaptive", "ppo_kl_per_sample"],
)
def test_each_variant_writes_a_valid_run_dir(kind: str, tmp_artifacts):
    data = apply_overrides(
        _cartpole_dict(kind),
        [
            "trainer.total_steps=256",
            "rollout.num_envs=2",
            "rollout.n_steps=32",
            "rollout.async_envs=false",
            "algorithm.minibatch_size=16",
            "algorithm.epochs=1",
            "logging.parquet=false",
            "logging.stdout=false",
        ],
    )
    cfg = ExperimentConfig.model_validate(data)
    result = worker_main(cfg, seed=0, allow_overwrite=True)
    run_dir = result.run_dir

    assert (run_dir / "snapshot.json").exists()
    assert (run_dir / "metadata.json").exists()
    events_path = run_dir / "logs" / "events.jsonl"
    assert events_path.exists()
    events = [
        json.loads(line)["event"] for line in events_path.read_text().splitlines()
    ]
    assert "run_created" in events
    assert "run_finished" in events
    assert (run_dir / "checkpoints" / "final.pt").exists()
    assert result.exit_status == "ok"
    assert result.iterations >= 1


def test_diagnostic_mode_full_writes_epoch_artifact(tmp_artifacts):
    """Full diagnostic mode emits ``metrics/epochs.parquet`` with migration columns."""
    data = apply_overrides(
        _cartpole_soft_dict("linear_ramp"),
        [
            "trainer.total_steps=512",
            "trainer.diagnostic_mode=full",
            "rollout.num_envs=2",
            "rollout.n_steps=32",
            "rollout.async_envs=false",
            "algorithm.minibatch_size=16",
            "algorithm.epochs=2",
            "logging.parquet=true",
            "logging.stdout=false",
        ],
    )
    cfg = ExperimentConfig.model_validate(data)
    result = worker_main(cfg, seed=0, allow_overwrite=True)

    assert result.exit_status == "ok"

    import pyarrow.parquet as pq

    epoch_path = result.run_dir / "metrics" / "epochs.parquet"
    assert epoch_path.exists(), "diagnostic_mode=full should emit epochs.parquet"
    epoch_rows = pq.read_table(epoch_path).to_pylist()
    assert len(epoch_rows) >= 2, "expected at least one (iteration, epoch) pair logged"
    # Epoch 0 has no prior epoch within its iteration; migration_rate is null.
    by_epoch = {(r["time/iteration"], r["epoch/index"]): r for r in epoch_rows}
    first_iter = min(it for it, _ in by_epoch)
    assert by_epoch[(first_iter, 0)]["epoch/migration/rate"] is None
    # Epoch 1 of any iteration must have a numeric migration rate.
    assert by_epoch[(first_iter, 1)]["epoch/migration/rate"] is not None
    assert 0.0 <= by_epoch[(first_iter, 1)]["epoch/migration/rate"] <= 1.0
    assert by_epoch[(first_iter, 1)]["epoch/optim/policy_grad_norm/var"] is not None

    train_row = pq.read_table(result.run_dir / "metrics" / "train.parquet").to_pylist()[
        -1
    ]
    assert train_row["diagnostics/migration_rate/mean"] is not None
    assert train_row["diagnostics/migration_rate/max"] is not None
    assert train_row["diagnostics/policy_grad_norm_var_per_epoch/mean"] is not None


def test_per_sample_variant_writes_beta_quantiles(tmp_artifacts):
    """Per-sample KL logs rollout-level beta distribution quantiles."""
    data = apply_overrides(
        _cartpole_dict("ppo_kl_per_sample"),
        [
            "trainer.total_steps=64",
            "rollout.num_envs=2",
            "rollout.n_steps=32",
            "rollout.async_envs=false",
            "algorithm.minibatch_size=16",
            "algorithm.epochs=1",
            "logging.parquet=true",
            "logging.stdout=false",
        ],
    )
    cfg = ExperimentConfig.model_validate(data)
    result = worker_main(cfg, seed=0, allow_overwrite=True)

    assert result.exit_status == "ok"

    import pyarrow.parquet as pq

    row = pq.read_table(result.run_dir / "metrics" / "train.parquet").to_pylist()[-1]
    assert row["beta/per_sample/all/p01"] is not None
    assert row["beta/per_sample/all/p50"] is not None
    assert row["beta/per_sample/all/p99"] is not None
    assert row["beta/per_sample/I_kill/p50"] is None or isinstance(
        row["beta/per_sample/I_kill/p50"], float
    )
    assert row["beta/times_adv_abs/p50"] is not None


@pytest.mark.parametrize("method", ["linear_ramp", "sigmoid", "soft_min"])
def test_each_soft_clip_method_writes_metrics(method: str, tmp_artifacts):
    data = apply_overrides(
        _cartpole_soft_dict(method),
        [
            "trainer.total_steps=256",
            "rollout.num_envs=2",
            "rollout.n_steps=32",
            "rollout.async_envs=false",
            "algorithm.minibatch_size=16",
            "algorithm.epochs=1",
            "logging.parquet=true",
            "logging.stdout=false",
        ],
    )
    cfg = ExperimentConfig.model_validate(data)
    result = worker_main(cfg, seed=0, allow_overwrite=True)

    assert result.exit_status == "ok"
    assert result.iterations >= 1

    import pyarrow.parquet as pq

    row = pq.read_table(result.run_dir / "metrics" / "train.parquet").to_pylist()[-1]
    assert row["soft_clip/softness"] == pytest.approx(0.05)
    assert row["soft_clip/gate/mean/all"] is not None
    if method == "soft_min":
        assert row["soft_clip/unclipped_branch_weight/mean/all"] is not None
    else:
        assert row["soft_clip/effective_beta/abs_mean/all"] is not None


def test_normalized_run_checkpoints_normalizer_and_logs_periodic_eval(tmp_artifacts):
    data = apply_overrides(
        _cartpole_dict("ppo_clip"),
        [
            "trainer.total_steps=64",
            "trainer.eval_every_steps=64",
            "trainer.eval_episodes=2",
            "rollout.num_envs=2",
            "rollout.n_steps=32",
            "rollout.async_envs=false",
            "env.normalize_obs=true",
            "env.normalize_reward=true",
            "algorithm.minibatch_size=16",
            "algorithm.epochs=1",
            "logging.parquet=true",
            "logging.stdout=false",
        ],
    )
    cfg = ExperimentConfig.model_validate(data)
    result = worker_main(cfg, seed=0, allow_overwrite=True)
    run_dir = result.run_dir

    state = CheckpointManager(run_dir).load(run_dir / "checkpoints" / "final.pt")
    normalizer = state["collector"]["normalizer"]
    assert normalizer["normalize_obs"] is True
    assert normalizer["normalize_reward"] is True
    assert normalizer["gamma"] == cfg.algorithm.gamma

    import pyarrow.parquet as pq

    table = pq.read_table(run_dir / "metrics" / "train.parquet")
    row = table.to_pylist()[-1]
    assert row["eval/episode/count"] == 2
    assert row["eval/return/mean"] > 0.0
