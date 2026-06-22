"""PPO-KL with adaptive scalar β (§2.3 dual-ascent rule).

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms",
    arXiv:1707.06347, 2017, §2.3. This implements the paper's adaptive
    scalar KL coefficient rule, not the clipped surrogate.

β is held **constant** during all inner epochs of one rollout update,
and updated **once** after the rollout update completes — matching the
PPO paper. The update key (measured KL ``d``) is the mean of whichever
KL estimator was used in the loss (``cfg.kl_penalty``) recomputed under
the **post-update** policy in one no-grad pass over the full rollout;
this keeps the control variable consistent with the dual-ascent target
in §2.3 and avoids blurring the final policy movement with the path of
KL across inner-loop minibatch steps.

Rule (PPO paper, §2.3):

    d = KL(π_old || π_new) under the post-update policy
    if d > kl_high_ratio * kl_target:  β ← β * beta_inc_factor
    if d < kl_low_ratio  * kl_target:  β ← β / beta_inc_factor
"""

from __future__ import annotations

from typing import Any

import torch

from klip_ppo.configs.algorithm.ppo_kl_adaptive import PPOKLAdaptiveConfig
from klip_ppo.core.ppo.strategies.base import (
    BasePPOLossStrategy,
    EpochAggregate,
    _Shared,
    select_kl_penalty,
)
from klip_ppo.core.rollout import PPOMinibatch


class KLAdaptiveStrategy(BasePPOLossStrategy):
    name = "ppo_kl_adaptive"

    def __init__(self, cfg: PPOKLAdaptiveConfig) -> None:
        super().__init__(cfg)
        self.cfg: PPOKLAdaptiveConfig = cfg
        self._beta = float(cfg.beta_init)

    @property
    def beta(self) -> float:
        return self._beta

    def _policy_loss(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        beta = self._beta
        surrogate = shared.ratio * mb.advantages
        kl_t = select_kl_penalty(shared, self.cfg.kl_penalty)
        loss = -surrogate.mean() + beta * kl_t.mean()
        diag = {
            "beta": torch.as_tensor(beta, device=loss.device),
            "kl_penalty": (beta * kl_t.detach()).mean(),
        }
        return loss, diag

    def on_rollout_end(
        self,
        agg: EpochAggregate,
        *,
        final_kl: dict[str, float] | None = None,
    ) -> None:
        """
        Update β using the selected post-update KL estimator.

        Reads the mean KL between rollout policy and post-update policy from
        ``final_kl[kind]``, where ``kind = cfg.kl_penalty``. The trainer recomputes
        ``final_kl`` once per rollout update in one no-grad pass over the full
        rollout, so β is driven by the dual-ascent target ``d`` in Schulman 2017
        §2.3 rather than the rollout-aggregate KL collected while the policy was
        still moving. Raises if the matching estimator is missing rather than
        silently falling back to the during-training aggregate, which would change
        the adaptive rule.
        """
        kind = self.cfg.kl_penalty
        if final_kl is None or kind not in final_kl:
            raise RuntimeError(
                "adaptive-beta controller requires final_kl computed under the "
                f"post-update policy for kl_penalty={kind!r}; trainer did not "
                "pass it. This is a wiring bug, not a runtime condition."
            )
        kl = final_kl[kind]
        target = self.cfg.kl_target
        if kl > self.cfg.kl_high_ratio * target:
            self._beta = min(self._beta * self.cfg.beta_inc_factor, self.cfg.beta_max)
        elif kl < self.cfg.kl_low_ratio * target:
            self._beta = max(self._beta / self.cfg.beta_inc_factor, self.cfg.beta_min)

    def state_dict(self) -> dict[str, Any]:
        return {"beta": self._beta}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "beta" in state:
            self._beta = float(state["beta"])
