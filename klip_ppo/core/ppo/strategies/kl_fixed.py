"""PPO-KL with a fixed scalar β.

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms",
    arXiv:1707.06347, 2017, §2. The PPO paper presents this scalar
    KL-penalty objective alongside PPO-Clip.

Optimised objective (per the PPO paper, §2 with constant β):

    L = -E[w_t A_t] + β * E[ KL_penalty_t ]

where ``KL_penalty_t`` is whichever per-sample estimator the user
selected via ``cfg.kl_penalty``. Default is ``"full"`` =
``KL(π_old(·|s_t) || π_new(·|s_t))``, matching the literature.
"""

from __future__ import annotations

import torch

from klip_ppo.configs.algorithm.ppo_kl_fixed import PPOKLFixedConfig
from klip_ppo.core.ppo.strategies.base import (
    BasePPOLossStrategy,
    _Shared,
    select_kl_penalty,
)
from klip_ppo.core.rollout import PPOMinibatch


class KLFixedStrategy(BasePPOLossStrategy):
    name = "ppo_kl_fixed"

    def __init__(self, cfg: PPOKLFixedConfig) -> None:
        super().__init__(cfg)
        self.cfg: PPOKLFixedConfig = cfg

    def _policy_loss(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        beta = self.cfg.beta
        surrogate = shared.ratio * mb.advantages
        kl_t = select_kl_penalty(shared, self.cfg.kl_penalty)
        loss = -surrogate.mean() + beta * kl_t.mean()
        diag = {
            "beta": torch.as_tensor(beta, device=loss.device),
            "kl_penalty": (beta * kl_t.detach()).mean(),
        }
        return loss, diag
