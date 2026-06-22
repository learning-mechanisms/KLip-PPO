"""PPO-KL with per-sample β (§3.5 of the local paper).

Reference context:
    This is a repo-local reformulation of PPO-Clip, not the scalar
    PPO-KL penalty in Schulman et al., "Proximal Policy Optimization
    Algorithms", arXiv:1707.06347, 2017.

Derivation (sketch). PPO-Clip's gradient w.r.t. θ_new is

    ∂L_clip / ∂θ = -A_t * ∂w_t / ∂θ        for t ∈ I_in ∪ I_pass
                 =  0                       for t ∈ I_kill .

The per-sample β formulation reproduces this gradient exactly by adding
to the unclipped surrogate a single-sample log-probability penalty

    kl_t = -log w_t = log π_old(a_t|s_t) - log π_new(a_t|s_t)

weighted by a *detached* per-sample coefficient

    β_t  =  -w_t * A_t       (t ∈ I_kill)
    β_t  =   0               (otherwise)

giving the loss ``-E[w_t A_t] + E[β_t * kl_t]``. Its gradient cancels
the unclipped surrogate term wherever I_kill triggers and leaves the
other samples untouched, matching PPO-Clip pointwise. This is the
equivalence proof; the test in ``tests/integration/test_equivalence.py``
verifies it numerically.

IMPORTANT: do not "fix" this strategy to use the full closed-form KL.
The equivalence proof is about cancelling the *sampled* surrogate
gradient with the *sampled* log-probability penalty. Substituting the
analytic ``KL(π_old || π_new)`` breaks the cancellation.
"""

from __future__ import annotations

import torch

from klip_ppo.configs.algorithm.ppo_kl_per_sample import PPOKLPerSampleConfig
from klip_ppo.core.ppo.strategies.base import BasePPOLossStrategy, _Shared
from klip_ppo.core.rollout import PPOMinibatch


class KLPerSampleStrategy(BasePPOLossStrategy):
    name = "ppo_kl_per_sample"

    def __init__(self, cfg: PPOKLPerSampleConfig) -> None:
        super().__init__(cfg)
        self.cfg: PPOKLPerSampleConfig = cfg

    def _policy_loss(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ratio = shared.ratio
        adv = mb.advantages
        kill_mask = shared.partition.kill
        beta_t = _per_sample_beta(ratio, adv, kill_mask)
        surrogate = ratio * adv
        # Per-sample log-probability KL surrogate, NOT the full KL.
        kl_t = -shared.log_ratio
        loss = -surrogate.mean() + (beta_t * kl_t).mean()

        beta_det = beta_t.detach()
        zero = torch.zeros((), device=loss.device, dtype=beta_det.dtype)
        if kill_mask.any():
            beta_abs_kill = beta_det.abs()[kill_mask].mean()
            beta_signed_kill = beta_det[kill_mask].mean()
        else:
            beta_abs_kill = zero
            beta_signed_kill = zero
        diag = {
            "beta_abs_mean_all": beta_det.abs().mean(),
            "beta_abs_mean_I_kill": beta_abs_kill,
            "beta_signed_mean_I_kill": beta_signed_kill,
            "kl_penalty": (beta_det * kl_t.detach()).mean(),
            "frac_in_I_kill_exact": kill_mask.float().mean(),
        }
        return loss, diag

    def _sample_diagnostics(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> dict[str, torch.Tensor]:
        kill_mask = shared.partition.kill
        beta_t = _per_sample_beta(shared.ratio, mb.advantages, kill_mask)
        adv_det = mb.advantages.detach()
        return {
            "beta/per_sample/all": beta_t,
            "beta/per_sample/I_kill": beta_t[kill_mask],
            "beta/times_adv_abs": (beta_t * adv_det).abs(),
        }


def _per_sample_beta(
    ratio: torch.Tensor, adv: torch.Tensor, kill_mask: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        return torch.where(
            kill_mask,
            -ratio.detach() * adv.detach(),
            torch.zeros_like(ratio),
        )
