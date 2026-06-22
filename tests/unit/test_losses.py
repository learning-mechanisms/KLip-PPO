"""Unit tests for PPO-Clip primitives."""

from __future__ import annotations

import pytest
import torch
from klip_ppo.core.losses import (
    approx_kl_from_logratio,
    clipped_surrogate,
    clipped_value_loss,
    harmful_clip_distance,
    partition_indices,
    policy_ratio,
    soft_clip_gate_linear,
    soft_clip_gate_sigmoid,
    soft_clipped_surrogate_softmin,
)


def test_partition_assigns_each_nonzero_advantage_sample_exactly_once():
    """
    For nonzero advantages, I_in, I_pass, I_kill partition the batch.

    Zero-advantage samples are not assigned to any of the three named classes when their
    ratio is out of band; see the ``frac_in_I_unclassified`` diagnostic on
    ``BasePPOLossStrategy``.
    """
    ratio = torch.tensor([1.30, 1.00, 0.70, 1.30, 0.70])
    adv = torch.tensor([+1.0, +1.0, +1.0, -1.0, -1.0])
    part = partition_indices(ratio, adv, clip_epsilon=0.2)
    union = part.in_band | part.pass_through | part.kill
    assert union.all()
    assert not (part.in_band & part.pass_through).any()
    assert not (part.in_band & part.kill).any()
    assert not (part.pass_through & part.kill).any()
    # Hand-checked classes:
    expected_kill = torch.tensor([True, False, False, False, True])
    expected_pass = torch.tensor([False, False, True, True, False])
    expected_in = torch.tensor([False, True, False, False, False])
    assert torch.equal(part.kill, expected_kill)
    assert torch.equal(part.pass_through, expected_pass)
    assert torch.equal(part.in_band, expected_in)


def test_partition_leaves_zero_advantage_out_of_band_unclassified():
    """
    A sample with adv=0 and ratio outside [1-eps, 1+eps] sits in no class.

    This is the documented behaviour; the trainer logs the residual as
    ``frac_in_I_unclassified``. The test pins the convention so we notice if it changes
    accidentally.
    """
    ratio = torch.tensor([1.30, 0.70, 1.00])
    adv = torch.tensor([0.0, 0.0, 0.0])
    part = partition_indices(ratio, adv, clip_epsilon=0.2)
    assert not part.kill.any()
    assert not part.pass_through.any()
    # ratio=1.0 is in band; the others are not.
    assert torch.equal(part.in_band, torch.tensor([False, False, True]))


def test_clipped_surrogate_matches_min_formula():
    ratio = torch.tensor([1.30, 0.70])
    adv = torch.tensor([+1.0, +1.0])
    loss, clip_fraction = clipped_surrogate(ratio, adv, clip_epsilon=0.2)
    # ratio=1.3, A=1: min(1.3, 1.2) = 1.2 → contributes -1.2
    # ratio=0.7, A=1: min(0.7, 0.8) = 0.7 → contributes -0.7
    expected = -((1.2 + 0.7) / 2)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)
    assert torch.allclose(clip_fraction, torch.tensor(1.0))


def test_harmful_clip_distance_is_sign_aware():
    ratio = torch.tensor([1.30, 0.70, 0.70, 1.30, 1.00, 1.30])
    adv = torch.tensor([+1.0, -1.0, +1.0, -1.0, +1.0, 0.0])
    distance = harmful_clip_distance(ratio, adv, clip_epsilon=0.2)
    expected = torch.tensor([0.10, 0.10, -0.50, -0.50, -0.20, float("-inf")])
    assert torch.allclose(distance, expected)


def test_soft_clip_linear_gate_matches_ramp_regions():
    ratio = torch.tensor([1.10, 0.70, 1.25, 1.30, 0.75, 0.70, 1.30])
    adv = torch.tensor([+1.0, +1.0, +1.0, +1.0, -1.0, -1.0, 0.0])
    gate = soft_clip_gate_linear(ratio, adv, clip_epsilon=0.2, softness=0.1)
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 0.5, 1.0, 0.0])
    assert torch.allclose(gate, expected, atol=1e-6)


def test_soft_clip_sigmoid_gate_is_monotone_and_temperature_controlled():
    adv = torch.ones(5)
    ratio = torch.tensor([1.00, 1.10, 1.20, 1.25, 1.40])
    gate_soft = soft_clip_gate_sigmoid(ratio, adv, clip_epsilon=0.2, softness=0.1)
    gate_sharp = soft_clip_gate_sigmoid(ratio, adv, clip_epsilon=0.2, softness=0.01)

    assert torch.all(gate_soft[1:] >= gate_soft[:-1])
    assert gate_soft[2].item() == pytest.approx(0.5)
    assert gate_sharp[0] < gate_soft[0]
    assert gate_sharp[-1] > gate_soft[-1]


def test_soft_gate_tiny_softness_has_hard_clip_gradient():
    ratio = torch.tensor([1.50, 0.50, 0.60, 1.40, 1.05], requires_grad=True)
    adv = torch.tensor([+1.0, -1.0, +1.0, -1.0, +0.5])
    hard_loss, _ = clipped_surrogate(ratio, adv, clip_epsilon=0.2)
    hard_grad = torch.autograd.grad(hard_loss, ratio)[0]

    for gate_fn in (soft_clip_gate_linear, soft_clip_gate_sigmoid):
        ratio_soft = ratio.detach().clone().requires_grad_(True)
        gate = gate_fn(
            ratio_soft.detach(), adv, clip_epsilon=0.2, softness=1e-4
        ).detach()
        beta_t = -ratio_soft.detach() * adv * gate
        soft_loss = -(ratio_soft * adv).mean() + (beta_t * (-ratio_soft.log())).mean()
        soft_grad = torch.autograd.grad(soft_loss, ratio_soft)[0]
        assert torch.allclose(soft_grad, hard_grad, atol=1e-5)


def test_soft_gate_keeps_partial_gradient_just_outside_kill_region():
    ratio = torch.tensor([1.21], requires_grad=True)
    adv = torch.tensor([1.0])
    gate = soft_clip_gate_linear(
        ratio.detach(), adv, clip_epsilon=0.2, softness=0.1
    ).detach()
    beta_t = -ratio.detach() * adv * gate
    loss = -(ratio * adv).mean() + (beta_t * (-ratio.log())).mean()
    grad = torch.autograd.grad(loss, ratio)[0]

    assert gate.item() == pytest.approx(0.1, abs=1e-6)
    assert grad.item() == pytest.approx(-0.9, abs=1e-6)


def test_softmin_tiny_softness_matches_clipped_surrogate_gradient():
    ratio = torch.tensor([1.50, 0.50, 0.60, 1.40, 1.05], requires_grad=True)
    adv = torch.tensor([+1.0, -1.0, +1.0, -1.0, +0.5])
    hard_loss, _ = clipped_surrogate(ratio, adv, clip_epsilon=0.2)
    hard_grad = torch.autograd.grad(hard_loss, ratio)[0]

    ratio_soft = ratio.detach().clone().requires_grad_(True)
    soft_loss, branch_weight, _ = soft_clipped_surrogate_softmin(
        ratio_soft, adv, clip_epsilon=0.2, softness=1e-4
    )
    soft_grad = torch.autograd.grad(soft_loss, ratio_soft)[0]

    assert torch.allclose(soft_loss, hard_loss, atol=1e-5)
    assert torch.allclose(soft_grad, hard_grad, atol=1e-4)
    assert torch.all((branch_weight >= 0.0) & (branch_weight <= 1.0))


def test_softmin_is_below_hard_min_and_handles_zero_advantages():
    ratio = torch.tensor([1.15, 1.25, 0.75, 1.30])
    adv = torch.tensor([1.0, 1.0, -1.0, 0.0])
    loss, branch_weight, _ = soft_clipped_surrogate_softmin(
        ratio, adv, clip_epsilon=0.2, softness=0.05
    )
    soft_obj = -loss * ratio.numel()
    hard_obj = torch.minimum(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).sum()

    assert torch.isfinite(loss)
    assert soft_obj <= hard_obj + 1e-6
    assert branch_weight[-1].item() == 0.0


def test_policy_ratio_equals_exp_log_diff():
    a = torch.tensor([-0.5, -1.2, 0.1])
    b = torch.tensor([-1.0, -1.0, 0.0])
    assert torch.allclose(policy_ratio(a, b), torch.exp(a - b), atol=1e-7)


def test_approx_kl_nonneg_in_expectation():
    torch.manual_seed(0)
    log_ratio = torch.randn(1000) * 0.05
    assert float(approx_kl_from_logratio(log_ratio)) >= 0.0


def test_clipped_value_loss_falls_back_to_mse_when_unclipped():
    values = torch.tensor([0.5, -0.5])
    old = torch.tensor([0.0, 0.0])
    ret = torch.tensor([1.0, -1.0])
    plain = clipped_value_loss(values, old, ret, clip_epsilon=0.2, clip=False)
    # 0.5 * mean((0.5-1)**2 + (-0.5 - -1)**2) = 0.5 * mean(0.25, 0.25) = 0.125
    assert torch.allclose(plain, torch.tensor(0.125), atol=1e-6)
