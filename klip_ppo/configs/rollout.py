"""Rollout collection configuration."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from klip_ppo.configs.base import BaseConfig


class RolloutConfig(BaseConfig):
    """
    How rollouts are collected each training iteration.

    The global batch per iteration is ``num_envs * n_steps``. The Job runs all envs in
    this Job process (sync) or in subprocess workers of this Job process (async).
    """

    num_envs: Annotated[int, Field(gt=0)]
    """Number of vectorized environments collected in parallel."""

    n_steps: Annotated[int, Field(gt=0)]
    """Number of environment steps collected per environment per rollout."""

    async_envs: bool = False
    """Whether to run vectorized environments in subprocess workers."""
