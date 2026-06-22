"""Training metric aggregation tests."""

from __future__ import annotations

import pytest
import torch
from klip_ppo.configs.algorithm.ppo_soft_clip import PPOSoftClipConfig, SoftClipMethod
from klip_ppo.core.distributions import PolicyDistParams
from klip_ppo.core.losses import partition_indices
from klip_ppo.core.ppo.diagnostic_metrics import BETA_QUANTILE_KEYS, BETA_QUANTILES
from klip_ppo.core.ppo.strategies.base import EpochAggregate, StrategyOutputs, _Shared
from klip_ppo.core.ppo.strategies.soft_clip import SoftClipStrategy
from klip_ppo.core.ppo.trainer import (
    PPOTrainer,
    RolloutTrainStats,
    _summarise_sample_quantiles,
)
from klip_ppo.core.rollout import EpisodeStats, PPOMinibatch
from klip_ppo.utils.logging import PARQUET_SCHEMA


def test_epoch_aggregate_preserves_ratio_extrema_and_means_percentiles() -> None:
    agg = EpochAggregate()
    agg.update(_outputs(ratio_min=0.8, ratio_max=1.2, ratio_p05=0.9, ratio_p95=1.1))
    agg.update(_outputs(ratio_min=0.5, ratio_max=1.8, ratio_p05=0.7, ratio_p95=1.4))

    metrics = agg.as_dict()

    assert agg.counts == 2
    assert metrics["ratio_min"] == pytest.approx(0.5)
    assert metrics["ratio_max"] == pytest.approx(1.8)
    assert metrics["ratio_p05"] == pytest.approx(0.8)
    assert metrics["ratio_p95"] == pytest.approx(1.25)


def test_trainer_log_row_exposes_canonical_metrics() -> None:
    agg = EpochAggregate()
    agg.update(
        _outputs(
            approx_kl=0.01,
            kl_full_mean=0.02,
            kl_sample_mean=0.015,
            ratio_mean=1.0,
            ratio_min=0.8,
            ratio_max=1.2,
            ratio_p05=0.9,
            ratio_p95=1.1,
            soft_clip_softness=0.05,
            soft_clip_gate_mean=0.25,
            soft_clip_gate_mean_I_kill=0.75,
        ),
        policy_grad_norm=2.0,
    )
    agg.update(
        _outputs(
            approx_kl=0.01,
            kl_full_mean=0.02,
            kl_sample_mean=0.015,
            ratio_mean=1.0,
            ratio_min=0.8,
            ratio_max=1.2,
            ratio_p05=0.9,
            ratio_p95=1.1,
            soft_clip_softness=0.05,
            soft_clip_gate_mean=0.25,
            soft_clip_gate_mean_I_kill=0.75,
        ),
        policy_grad_norm=4.0,
    )
    ep_stats = EpisodeStats(
        raw_returns=[10.0, 20.0],
        wrapped_returns=[1.0, 2.0],
        lengths=[50.0, 60.0],
    )
    trainer = object.__new__(PPOTrainer)
    trainer.optim = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr=0.123)

    row = trainer._build_log_row(
        iteration=1,
        env_step=128,
        wall_s=2.5,
        train_stats=RolloutTrainStats(
            aggregate=agg,
            early_stopped=True,
            sample_quantiles={"beta/per_sample/all/p50": -0.25},
        ),
        explained_var=0.75,
        ep_stats=ep_stats,
    )

    assert row["policy/kl/full_mean"] == pytest.approx(0.02)
    assert row["policy/kl/sample_mean"] == pytest.approx(0.015)
    assert row["policy/ratio/min"] == pytest.approx(0.8)
    assert row["policy/ratio/max"] == pytest.approx(1.2)
    assert row["policy/ratio/p05"] == pytest.approx(0.9)
    assert row["policy/ratio/p95"] == pytest.approx(1.1)
    assert row["soft_clip/softness"] == pytest.approx(0.05)
    assert row["soft_clip/gate/mean/all"] == pytest.approx(0.25)
    assert row["soft_clip/gate/mean/I_kill"] == pytest.approx(0.75)
    assert row["optim/policy_grad_norm/mean"] == pytest.approx(3.0)
    assert row["optim/policy_grad_norm/var"] == pytest.approx(1.0)
    assert row["optim/policy_grad_norm/std"] == pytest.approx(1.0)
    assert row["train/episode/count"] == 2
    assert row["update/steps"] == 2
    assert row["update/early_stopped"] == pytest.approx(1.0)
    assert row["optim/lr"] == pytest.approx(0.123)
    assert row["beta/per_sample/all/p50"] == pytest.approx(-0.25)


def test_sample_quantiles_cover_beta_distribution_metrics() -> None:
    metrics = _summarise_sample_quantiles(
        {
            "beta/per_sample/all": [
                torch.tensor([0.0, -2.0]),
                torch.tensor([1.0, 3.0]),
            ],
            "beta/per_sample/I_kill": [torch.tensor([-2.0, 1.0, 3.0])],
            "beta/times_adv_abs": [torch.tensor([0.0, 4.0, 2.0, 6.0])],
        }
    )

    assert metrics["beta/per_sample/all/p01"] == pytest.approx(-1.94)
    assert metrics["beta/per_sample/all/p50"] == pytest.approx(0.5)
    assert metrics["beta/per_sample/all/p99"] == pytest.approx(2.94)
    assert metrics["beta/per_sample/I_kill/p50"] == pytest.approx(1.0)
    assert metrics["beta/times_adv_abs/p95"] == pytest.approx(5.7)


def test_sample_quantiles_mark_empty_kill_distribution_null() -> None:
    metrics = _summarise_sample_quantiles(
        {"beta/per_sample/I_kill": [torch.tensor([])]}
    )

    assert metrics["beta/per_sample/I_kill/p01"] is None
    assert metrics["beta/per_sample/I_kill/p99"] is None


def test_beta_quantile_metrics_are_in_parquet_schema() -> None:
    for key in BETA_QUANTILE_KEYS:
        assert PARQUET_SCHEMA[key] == "float64"


@pytest.mark.parametrize("method", ["linear_ramp", "sigmoid", "soft_min"])
def test_soft_clip_emits_per_sample_beta_diagnostics(method: SoftClipMethod) -> None:
    cfg = PPOSoftClipConfig(method=method, clip_epsilon=0.2, softness=0.05)
    strategy = SoftClipStrategy(cfg)
    mb, shared = _soft_clip_fixture(clip_epsilon=cfg.clip_epsilon)

    samples = strategy._sample_diagnostics(mb, shared)

    assert set(samples) == {
        "beta/per_sample/all",
        "beta/per_sample/I_kill",
        "beta/times_adv_abs",
    }
    assert samples["beta/per_sample/all"].shape == mb.advantages.shape
    assert samples["beta/times_adv_abs"].shape == mb.advantages.shape
    assert samples["beta/per_sample/I_kill"].shape == (
        int(shared.partition.kill.sum()),
    )
    for tensor in samples.values():
        assert torch.isfinite(tensor).all()
    assert (samples["beta/times_adv_abs"] >= 0).all()


def test_soft_clip_sample_diagnostics_feed_full_quantile_keys() -> None:
    cfg = PPOSoftClipConfig(method="linear_ramp", clip_epsilon=0.2, softness=0.05)
    strategy = SoftClipStrategy(cfg)
    mb, shared = _soft_clip_fixture(clip_epsilon=cfg.clip_epsilon)

    samples = strategy._sample_diagnostics(mb, shared)
    metrics = _summarise_sample_quantiles(
        {key: [tensor.detach().cpu()] for key, tensor in samples.items()}
    )

    for key in BETA_QUANTILE_KEYS:
        assert key in metrics
    for label, _ in BETA_QUANTILES:
        assert metrics[f"beta/per_sample/all/{label}"] is not None
        assert metrics[f"beta/times_adv_abs/{label}"] is not None


def _soft_clip_fixture(*, clip_epsilon: float) -> tuple[PPOMinibatch, _Shared]:
    """Hand-built ratio/advantage tensors with samples in every partition."""
    ratio = torch.tensor([1.00, 0.85, 1.15, 1.40, 0.60, 1.30, 0.70, 1.05])
    adv = torch.tensor([0.50, -0.40, 0.30, 0.80, -0.60, -0.20, 0.10, 0.00])
    partition = partition_indices(ratio, adv, clip_epsilon)
    # Sanity: the fixture must populate I_kill so the masked tensor is non-empty.
    assert partition.kill.any()
    log_ratio = ratio.log()
    placeholder = torch.zeros_like(ratio)
    shared = _Shared(
        new_logprobs=placeholder,
        new_dist_params=PolicyDistParams(kind="categorical", logits=placeholder),
        entropy=placeholder,
        values=placeholder,
        log_ratio=log_ratio,
        ratio=ratio,
        approx_kl=torch.tensor(0.0),
        value_loss=torch.tensor(0.0),
        partition=partition,
        full_kl_t=placeholder,
        sample_kl_t=placeholder,
        k3_kl_t=placeholder,
    )
    mb = PPOMinibatch(
        obs=placeholder,
        actions=placeholder,
        old_logprobs=placeholder,
        old_values=placeholder,
        advantages=adv,
        returns=placeholder,
        old_dist_params=PolicyDistParams(kind="categorical", logits=placeholder),
    )
    return mb, shared


def _outputs(**diagnostics: float) -> StrategyOutputs:
    zero = torch.tensor(0.0)
    return StrategyOutputs(
        policy_loss=zero,
        value_loss=zero,
        entropy=zero,
        total_loss=zero,
        diagnostics={k: torch.tensor(v) for k, v in diagnostics.items()},
    )
