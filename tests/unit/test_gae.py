"""Unit test for GAE against a hand-computed reference."""

from __future__ import annotations

import torch
from klip_ppo.core.gae import compute_gae


def test_gae_two_step_two_env_matches_hand_calc():
    # Two envs, two steps, no terminations. Hand-compute expected GAE.
    rewards = torch.tensor([[1.0, 0.5], [0.0, 2.0]])
    values = torch.tensor([[0.5, 1.0], [0.25, 0.5]])
    terminated = torch.tensor([[False, False], [False, False]])
    truncated = torch.tensor([[False, False], [False, False]])
    truncated_final_values = torch.zeros_like(values)
    final_value = torch.tensor([0.1, 0.2])
    gamma = 0.99
    gae_lambda = 0.95

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        terminated=terminated,
        truncated=truncated,
        final_value=final_value,
        truncated_final_values=truncated_final_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # δ₁ = r₁ + γ * V_final - V₁
    delta1 = rewards[1] + gamma * final_value - values[1]
    # A₁ = δ₁
    expected_adv1 = delta1
    # δ₀ = r₀ + γ * V₁ - V₀
    delta0 = rewards[0] + gamma * values[1] - values[0]
    # A₀ = δ₀ + γ λ A₁
    expected_adv0 = delta0 + gamma * gae_lambda * expected_adv1

    assert torch.allclose(advantages[0], expected_adv0, atol=1e-6)
    assert torch.allclose(advantages[1], expected_adv1, atol=1e-6)
    assert torch.allclose(returns, advantages + values, atol=1e-6)


def test_gae_terminal_resets_bootstrap():
    # If done at step 0, the bootstrap V_next must be zeroed out at step 0.
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.0]])
    terminated = torch.tensor([[True]])
    truncated = torch.tensor([[False]])
    final_value = torch.tensor([5.0])
    advantages, _ = compute_gae(
        rewards,
        values,
        terminated,
        truncated,
        final_value,
        torch.zeros_like(values),
        0.99,
        0.95,
    )
    # δ = r + γ * 0 * V_next - V = 1.0
    assert torch.allclose(advantages, torch.tensor([[1.0]]), atol=1e-6)


def test_gae_time_limit_truncation_bootstraps_final_observation():
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.0]])
    terminated = torch.tensor([[False]])
    truncated = torch.tensor([[True]])
    final_value = torch.tensor([99.0])
    truncated_final_values = torch.tensor([[5.0]])

    advantages, returns = compute_gae(
        rewards,
        values,
        terminated,
        truncated,
        final_value,
        truncated_final_values,
        0.99,
        0.95,
    )

    expected = torch.tensor([[1.0 + 0.99 * 5.0]])
    assert torch.allclose(advantages, expected, atol=1e-6)
    assert torch.allclose(returns, expected, atol=1e-6)
