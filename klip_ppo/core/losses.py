"""Pure functions used by all PPO loss strategies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClipPartition:
    """Boolean masks for the I_in / I_pass / I_kill partition from the paper.

    Definitions (with eps = ``clip_epsilon`` and ratio w_t):
      - I_in:   w_t ∈ [1-eps, 1+eps]                 (clip never triggers)
      - I_kill: (A_t > 0 and w_t > 1+eps) or
                (A_t < 0 and w_t < 1-eps)            (PPO-Clip kills the gradient)
      - I_pass: (A_t > 0 and w_t < 1-eps) or
                (A_t < 0 and w_t > 1+eps)            (gradient still flows)
    """

    in_band: torch.Tensor
    pass_through: torch.Tensor
    kill: torch.Tensor

    @property
    def frac_in(self) -> torch.Tensor:
        return self.in_band.float().mean()

    @property
    def frac_pass(self) -> torch.Tensor:
        return self.pass_through.float().mean()

    @property
    def frac_kill(self) -> torch.Tensor:
        return self.kill.float().mean()


def policy_ratio(
    new_logprobs: torch.Tensor, old_logprobs: torch.Tensor
) -> torch.Tensor:
    """Importance ratio ``w_t = π_new(a_t|s_t) / π_old(a_t|s_t)``."""
    return torch.exp(new_logprobs - old_logprobs)


def partition_indices(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_epsilon: float
) -> ClipPartition:
    """Return the I_in / I_pass / I_kill boolean masks."""
    eps = clip_epsilon
    above = ratio > 1.0 + eps
    below = ratio < 1.0 - eps
    pos = advantages > 0.0
    neg = advantages < 0.0
    in_band = ~(above | below)
    kill = (pos & above) | (neg & below)
    pass_through = (pos & below) | (neg & above)
    return ClipPartition(in_band=in_band, pass_through=pass_through, kill=kill)


def clipped_surrogate(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PPO-Clip surrogate.

    Returns ``(loss, clip_fraction)``.
    """
    eps = clip_epsilon
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    clip_fraction = ((ratio - 1.0).abs() > eps).float().mean()
    return loss, clip_fraction


def harmful_clip_distance(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """
    Signed distance to the harmful PPO-Clip boundary in ratio units.

    Positive values are exactly the ``I_kill`` region: ratios above
    ``1 + eps`` for positive advantages, and below ``1 - eps`` for
    negative advantages. Pass-through samples remain negative even when
    they are outside the clip band.
    """
    eps = clip_epsilon
    pos = advantages > 0.0
    neg = advantages < 0.0
    neg_inf = torch.full_like(ratio, float("-inf"))
    return torch.where(
        pos,
        ratio - (1.0 + eps),
        torch.where(neg, (1.0 - eps) - ratio, neg_inf),
    )


def soft_clip_gate_linear(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    softness: float,
) -> torch.Tensor:
    """Linear-ramp relaxation of the hard ``I_kill`` gate."""
    if softness <= 0.0:
        raise ValueError(f"softness must be > 0, got {softness!r}")
    distance = harmful_clip_distance(ratio, advantages, clip_epsilon)
    gate = (distance / softness).clamp(0.0, 1.0)
    return torch.where(advantages != 0.0, gate, torch.zeros_like(gate))


def soft_clip_gate_sigmoid(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    softness: float,
) -> torch.Tensor:
    """Sigmoid relaxation of the hard ``I_kill`` gate."""
    if softness <= 0.0:
        raise ValueError(f"softness must be > 0, got {softness!r}")
    distance = harmful_clip_distance(ratio, advantages, clip_epsilon)
    logits = (distance / softness).clamp(-60.0, 60.0)
    gate = torch.sigmoid(logits)
    return torch.where(advantages != 0.0, gate, torch.zeros_like(gate))


def soft_clipped_surrogate_softmin(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    softness: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    PPO-Clip surrogate with a normalized log-sum-exp soft minimum.

    ``softness`` is interpreted in ratio units by normalizing each sample's objective
    terms by ``abs(advantage)`` before applying the soft minimum. The returned branch
    weight is the derivative weight on the unclipped branch in the normalized soft-min
    objective.
    """
    if softness <= 0.0:
        raise ValueError(f"softness must be > 0, got {softness!r}")
    eps = clip_epsilon
    tau = softness
    adv_scale = advantages.abs().detach().clamp_min(1e-8)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
    unclipped_norm = unclipped / adv_scale
    clipped_norm = clipped / adv_scale
    stacked = torch.stack((-unclipped_norm / tau, -clipped_norm / tau), dim=0)
    soft_obj_norm = -tau * torch.logsumexp(stacked, dim=0)
    soft_obj = adv_scale * soft_obj_norm
    soft_obj = torch.where(advantages != 0.0, soft_obj, torch.zeros_like(soft_obj))

    branch_weight = torch.softmax(stacked, dim=0)[0]
    branch_weight = torch.where(
        advantages != 0.0, branch_weight, torch.zeros_like(branch_weight)
    )
    loss = -soft_obj.mean()
    diagnostics = {
        "soft_clip_unclipped_branch_weight_mean": branch_weight.detach().mean()
    }
    return loss, branch_weight, diagnostics


def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float,
    *,
    clip: bool,
) -> torch.Tensor:
    """
    Clipped MSE value loss (SB3 / Schulman 2017 convention).

    If ``clip`` is False, falls back to plain MSE.
    """
    if not clip:
        return 0.5 * (values - returns).pow(2).mean()
    clipped = old_values + (values - old_values).clamp(-clip_epsilon, clip_epsilon)
    loss_unclipped = (values - returns).pow(2)
    loss_clipped = (clipped - returns).pow(2)
    return 0.5 * torch.maximum(loss_unclipped, loss_clipped).mean()


def approx_kl_from_logratio(log_ratio: torch.Tensor) -> torch.Tensor:
    """John Schulman's low-variance KL estimate (k3): ``E[(r - 1) - log r]``."""
    ratio = log_ratio.exp()
    return ((ratio - 1.0) - log_ratio).mean()


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """``1 - Var[returns - values] / Var[returns]``; ``nan`` if returns are constant."""
    var = returns.var(unbiased=False)
    if var.item() < 1e-12:
        return torch.tensor(float("nan"), device=values.device)
    return 1.0 - (returns - values).var(unbiased=False) / var
