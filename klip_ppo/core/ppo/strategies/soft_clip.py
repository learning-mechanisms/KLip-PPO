"""Soft-clipped PPO strategies."""

from __future__ import annotations

import torch

from klip_ppo.configs.algorithm.ppo_soft_clip import PPOSoftClipConfig
from klip_ppo.core.losses import (
    soft_clip_gate_linear,
    soft_clip_gate_sigmoid,
    soft_clipped_surrogate_softmin,
)
from klip_ppo.core.ppo.strategies.base import BasePPOLossStrategy, _Shared
from klip_ppo.core.rollout import PPOMinibatch


class SoftClipStrategy(BasePPOLossStrategy):
    name = "ppo_soft_clip"

    def __init__(self, cfg: PPOSoftClipConfig) -> None:
        super().__init__(cfg)
        self.cfg: PPOSoftClipConfig = cfg

    def _policy_loss(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.cfg.method == "linear_ramp":
            gate = soft_clip_gate_linear(
                shared.ratio.detach(),
                mb.advantages,
                self.cfg.clip_epsilon,
                self.cfg.softness,
            )
            return self._gate_loss(mb, shared, gate)
        if self.cfg.method == "sigmoid":
            gate = soft_clip_gate_sigmoid(
                shared.ratio.detach(),
                mb.advantages,
                self.cfg.clip_epsilon,
                self.cfg.softness,
            )
            return self._gate_loss(mb, shared, gate)
        if self.cfg.method == "soft_min":
            return self._softmin_loss(mb, shared)
        raise ValueError(f"unknown soft clip method: {self.cfg.method!r}")

    def _gate_loss(
        self, mb: PPOMinibatch, shared: _Shared, gate: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ratio = shared.ratio
        adv = mb.advantages
        gate_det = gate.detach()
        with torch.no_grad():
            beta_t = -ratio.detach() * adv * gate_det
        surrogate = ratio * adv
        kl_t = -shared.log_ratio
        loss = -surrogate.mean() + (beta_t * kl_t).mean()
        diag = self._gate_diagnostics(
            gate_det,
            beta_t.detach(),
            shared,
            extra={"soft_clip_kl_penalty": (beta_t.detach() * kl_t.detach()).mean()},
        )
        return loss, diag

    def _softmin_loss(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, branch_weight, helper_diag = soft_clipped_surrogate_softmin(
            shared.ratio,
            mb.advantages,
            self.cfg.clip_epsilon,
            self.cfg.softness,
        )
        branch_det = branch_weight.detach()
        gate_det = self._softmin_effective_gate(branch_det, shared)
        diag = self._gate_diagnostics(gate_det, None, shared, extra=helper_diag)
        diag.update(
            {
                "soft_clip_unclipped_branch_weight_mean_I_in": _masked_mean(
                    branch_det, shared.partition.in_band
                ),
                "soft_clip_unclipped_branch_weight_mean_I_pass": _masked_mean(
                    branch_det, shared.partition.pass_through
                ),
                "soft_clip_unclipped_branch_weight_mean_I_kill": _masked_mean(
                    branch_det, shared.partition.kill
                ),
                "soft_clip_unclipped_branch_weight_mean_I_unclassified": _masked_mean(
                    branch_det, _unclassified_mask(shared)
                ),
            }
        )
        return loss, diag

    def _softmin_effective_gate(
        self, unclipped_branch_weight: torch.Tensor, shared: _Shared
    ) -> torch.Tensor:
        out_of_band = shared.partition.pass_through | shared.partition.kill
        gate = 1.0 - unclipped_branch_weight
        return torch.where(out_of_band, gate, torch.zeros_like(gate))

    def _effective_beta_t(self, mb: PPOMinibatch, shared: _Shared) -> torch.Tensor:
        """Detached per-sample β_t for whichever soft-clip method is active."""
        ratio_det = shared.ratio.detach()
        adv_det = mb.advantages.detach()
        if self.cfg.method == "linear_ramp":
            gate = soft_clip_gate_linear(
                ratio_det, adv_det, self.cfg.clip_epsilon, self.cfg.softness
            )
        elif self.cfg.method == "sigmoid":
            gate = soft_clip_gate_sigmoid(
                ratio_det, adv_det, self.cfg.clip_epsilon, self.cfg.softness
            )
        elif self.cfg.method == "soft_min":
            _, branch_weight, _ = soft_clipped_surrogate_softmin(
                shared.ratio,
                mb.advantages,
                self.cfg.clip_epsilon,
                self.cfg.softness,
            )
            gate = self._softmin_effective_gate(branch_weight.detach(), shared)
        else:
            raise ValueError(f"unknown soft clip method: {self.cfg.method!r}")
        return -ratio_det * adv_det * gate.detach()

    def _sample_diagnostics(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> dict[str, torch.Tensor]:
        beta_t = self._effective_beta_t(mb, shared)
        adv_det = mb.advantages.detach()
        return {
            "beta/per_sample/all": beta_t,
            "beta/per_sample/I_kill": beta_t[shared.partition.kill],
            "beta/times_adv_abs": (beta_t * adv_det).abs(),
        }

    def _gate_diagnostics(
        self,
        gate: torch.Tensor,
        beta_t: torch.Tensor | None,
        shared: _Shared,
        *,
        extra: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        diag: dict[str, torch.Tensor] = {
            "soft_clip_softness": torch.tensor(
                self.cfg.softness, device=gate.device, dtype=gate.dtype
            ),
            "soft_clip_gate_mean": gate.mean(),
            "soft_clip_gate_mean_I_in": _masked_mean(gate, shared.partition.in_band),
            "soft_clip_gate_mean_I_pass": _masked_mean(
                gate, shared.partition.pass_through
            ),
            "soft_clip_gate_mean_I_kill": _masked_mean(gate, shared.partition.kill),
            "soft_clip_gate_mean_I_unclassified": _masked_mean(
                gate, _unclassified_mask(shared)
            ),
        }
        if beta_t is not None:
            diag.update(
                {
                    "soft_clip_effective_beta_abs_mean_all": beta_t.abs().mean(),
                    "soft_clip_effective_beta_abs_mean_I_kill": _masked_mean(
                        beta_t.abs(), shared.partition.kill
                    ),
                    "soft_clip_effective_beta_signed_mean_I_kill": _masked_mean(
                        beta_t, shared.partition.kill
                    ),
                }
            )
        if extra is not None:
            diag.update(extra)
        return diag


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), device=values.device, dtype=values.dtype)
    if not bool(mask.any()):
        return zero
    return values[mask].mean()


def _unclassified_mask(shared: _Shared) -> torch.Tensor:
    return ~(
        shared.partition.in_band | shared.partition.pass_through | shared.partition.kill
    )
