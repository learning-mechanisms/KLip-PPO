"""
PPO-KL with adaptive scalar β (Schulman et al.

2017 dual-ascent rule).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from klip_ppo.configs.algorithm.base import PPOAlgoConfigBase


class PPOKLAdaptiveConfig(PPOAlgoConfigBase):
    """PPO-KL with adaptive β (§2.3 of the paper).

    β is updated **once per rollout update** (i.e. once after all
    ``epochs`` inner epochs over the rollout) using the literature
    rule from Schulman et al. 2017:

        d = mean KL over the just-completed rollout update
        if d > kl_high_ratio * kl_target:  β ← β * beta_inc_factor   (default ×2)
        if d < kl_low_ratio  * kl_target:  β ← β / beta_inc_factor   (default /2)
        otherwise                          β unchanged

    Defaults match the paper: ``kl_high_ratio=1.5``, ``kl_low_ratio=1/1.5``,
    ``beta_inc_factor=2``.

    The KL estimator used both in the loss and in the β update is
    chosen by ``kl_penalty`` (default ``"full"`` = closed-form
    ``KL(π_old || π_new)``).

    ``clip_epsilon_for_diagnostics`` is informational only.
    """

    kind: Literal["ppo_kl_adaptive"] = "ppo_kl_adaptive"
    """Discriminator selecting the adaptive-beta PPO-KL objective."""

    beta_init: Annotated[float, Field(gt=0.0)] = 1.0
    """Initial scalar KL penalty coefficient."""

    kl_target: Annotated[float, Field(gt=0.0)] = 0.02
    """Target mean KL used by the adaptive beta update."""

    # Default matches paper §4.1 (D_KL = 0.02); kept in sync with make_algorithm().

    beta_inc_factor: Annotated[float, Field(gt=1.0)] = 2.0
    """Multiplicative factor used to raise or lower beta."""

    beta_min: Annotated[float, Field(gt=0.0)] = 1e-4
    """Lower bound for the adaptive beta coefficient."""

    beta_max: Annotated[float, Field(gt=0.0)] = 1e4
    """Upper bound for the adaptive beta coefficient."""

    kl_low_ratio: Annotated[float, Field(gt=0.0, lt=1.0)] = 1.0 / 1.5
    """Target-relative KL ratio below which beta decreases."""

    kl_high_ratio: Annotated[float, Field(gt=1.0)] = 1.5
    """Target-relative KL ratio above which beta increases."""

    kl_penalty: Literal["full", "sample", "k3"] = "full"
    """Per-sample KL estimator used in the loss and beta update."""

    clip_epsilon_for_diagnostics: Annotated[float, Field(gt=0.0)] = 0.2
    """Clip radius used only for partition diagnostics."""
