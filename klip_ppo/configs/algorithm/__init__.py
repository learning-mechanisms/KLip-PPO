"""PPO algorithm configurations (discriminated by ``kind``)."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from klip_ppo.configs.algorithm.base import PPOAlgoConfigBase
from klip_ppo.configs.algorithm.ppo_clip import PPOClipConfig
from klip_ppo.configs.algorithm.ppo_kl_adaptive import PPOKLAdaptiveConfig
from klip_ppo.configs.algorithm.ppo_kl_fixed import PPOKLFixedConfig
from klip_ppo.configs.algorithm.ppo_kl_per_sample import PPOKLPerSampleConfig
from klip_ppo.configs.algorithm.ppo_soft_clip import PPOSoftClipConfig

AnyAlgorithmConfig = Annotated[
    PPOClipConfig
    | PPOKLFixedConfig
    | PPOKLAdaptiveConfig
    | PPOKLPerSampleConfig
    | PPOSoftClipConfig,
    Field(discriminator="kind"),
]

__all__ = [
    "AnyAlgorithmConfig",
    "PPOAlgoConfigBase",
    "PPOClipConfig",
    "PPOKLAdaptiveConfig",
    "PPOKLFixedConfig",
    "PPOKLPerSampleConfig",
    "PPOSoftClipConfig",
]
