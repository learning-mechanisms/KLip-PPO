"""
PPO-Clip strategy.

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms",
    arXiv:1707.06347, 2017. This is the clipped surrogate objective
    now commonly meant by "PPO-Clip".
"""

from __future__ import annotations

import torch

from klip_ppo.configs.algorithm.ppo_clip import PPOClipConfig
from klip_ppo.core.losses import clipped_surrogate
from klip_ppo.core.ppo.strategies.base import BasePPOLossStrategy, _Shared
from klip_ppo.core.rollout import PPOMinibatch


class ClipStrategy(BasePPOLossStrategy):
    name = "ppo_clip"

    def __init__(self, cfg: PPOClipConfig) -> None:
        super().__init__(cfg)
        self.cfg: PPOClipConfig = cfg

    def _policy_loss(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, clip_fraction = clipped_surrogate(
            shared.ratio, mb.advantages, self.cfg.clip_epsilon
        )
        return loss, {"clip_fraction_exact": clip_fraction.detach()}
