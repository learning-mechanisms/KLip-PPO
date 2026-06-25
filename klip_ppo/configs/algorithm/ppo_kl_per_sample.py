"""PPO-KL with per-sample adaptive β (Theorem 3.1, §3.5 of the paper)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from klip_ppo.configs.algorithm.base import PPOAlgoConfigBase


class PPOKLPerSampleConfig(PPOAlgoConfigBase):
    """PPO with the per-sample β derived in §3.5.

    For each sample ``t`` the effective KL coefficient is

        β_t = -w_t * Â_t        if  t ∈ I_kill
        β_t = 0                 otherwise

    where I_kill is the clip-suppressed set under ``clip_epsilon``.
    This reproduces the PPO-Clip gradient exactly (Theorem 3.1).
    """

    kind: Literal["ppo_kl_per_sample"] = "ppo_kl_per_sample"
    """Discriminator selecting the per-sample-beta PPO-KL objective."""

    clip_epsilon: Annotated[float, Field(gt=0.0)] = 0.2
    """Clip radius that defines pass/kill regions and per-sample beta."""
