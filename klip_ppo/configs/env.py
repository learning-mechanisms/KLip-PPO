"""Environment configuration."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from klip_ppo.configs.base import BaseConfig


class EnvConfig(BaseConfig):
    """Gymnasium environment configuration."""

    id: str
    """Gymnasium environment id passed to ``gym.make``."""

    normalize_obs: bool = False
    """Whether to normalize observations with running statistics."""

    normalize_reward: bool = False
    """Whether to normalize rewards with running return statistics."""

    clip_obs: Annotated[float, Field(gt=0.0)] = 10.0
    """Absolute clipping bound applied after observation normalization."""

    clip_reward: Annotated[float, Field(gt=0.0)] = 10.0
    """Absolute clipping bound applied after reward normalization."""

    reward_scale: Annotated[float, Field(gt=0.0)] = 1.0
    """Multiplier applied to rewards after optional normalization."""

    max_episode_steps: Annotated[int, Field(gt=0)] | None = None
    """Optional TimeLimit override for maximum episode length."""
