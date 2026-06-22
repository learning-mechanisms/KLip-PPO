"""PPO-Clip configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from klip_ppo.configs.algorithm.base import PPOAlgoConfigBase


class PPOClipConfig(PPOAlgoConfigBase):
    """
    PPO with the clipped surrogate objective (Schulman et al.

    2017, §1).
    """

    kind: Literal["ppo_clip"] = "ppo_clip"
    """Discriminator selecting the PPO-Clip objective."""

    clip_epsilon: Annotated[float, Field(gt=0.0)] = 0.2
    """Probability-ratio clipping radius used by the surrogate objective."""
