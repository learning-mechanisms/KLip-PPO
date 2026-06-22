"""Generalised Advantage Estimation (Schulman et al., 2016)."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    final_value: torch.Tensor,
    truncated_final_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages and returns from a (T, E) rollout.

    Args:
        rewards:     (T, E) per-step rewards.
        values:      (T, E) value-function estimates at the visited states.
        terminated:  (T, E) bool; True if the transition ended in a true terminal.
        truncated:   (T, E) bool; True if the transition ended due to a time limit.
        final_value: (E,)  value of the state reached after the last step.
        truncated_final_values:
                     (T, E) value estimates for final observations on time-limit
                     truncations; ignored where ``truncated`` is false.
        gamma:       discount factor.
        gae_lambda:  GAE-λ parameter.

    Returns:
        advantages, returns — both (T, E) tensors.
    """
    if (
        rewards.shape != values.shape
        or rewards.shape != terminated.shape
        or rewards.shape != truncated.shape
        or rewards.shape != truncated_final_values.shape
    ):
        raise ValueError(
            "shape mismatch: "
            f"rewards={rewards.shape} values={values.shape} "
            f"terminated={terminated.shape} truncated={truncated.shape} "
            f"truncated_final_values={truncated_final_values.shape}"
        )
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(final_value)
    for t in reversed(range(steps)):
        default_next_value = final_value if t == steps - 1 else values[t + 1]
        next_value = torch.where(
            truncated[t], truncated_final_values[t], default_next_value
        )
        bootstrap_mask = (~terminated[t]).to(rewards.dtype)
        continuation_mask = (~torch.logical_or(terminated[t], truncated[t])).to(
            rewards.dtype
        )
        delta = rewards[t] + gamma * next_value * bootstrap_mask - values[t]
        last_gae = delta + gamma * gae_lambda * continuation_mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
