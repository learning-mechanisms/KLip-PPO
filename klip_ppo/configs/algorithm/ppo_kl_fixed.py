"""PPO-KL with a fixed scalar penalty coefficient."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from klip_ppo.configs.algorithm.base import PPOAlgoConfigBase


class PPOKLFixedConfig(PPOAlgoConfigBase):
    """
    PPO with a fixed scalar KL penalty (§2 with constant β).

    The optimised objective is ``-E[w_t A_t] + β * E[ KL_penalty_t ]`` where
    ``KL_penalty_t`` is the per-sample estimator selected by ``kl_penalty``:

    - ``"full"``: closed-form ``KL(π_old(·|s_t) || π_new(·|s_t))``. This is
      what the PPO paper writes (and what TRPO inherits); it is the
      default. Implemented in ``klip_ppo.core.distributions.kl_old_new``.
    - ``"sample"``: single-sample estimator ``log π_old(a_t|s_t) - log π_new(a_t|s_t)``.
      Matches the form used in the local PDF's sample-based derivation.
    - ``"k3"``: John Schulman's low-variance estimator
      ``(w_t - 1) - log w_t``. Available for ablation; not silently used.

    ``clip_epsilon_for_diagnostics`` is retained only for the
    ``I_in / I_pass / I_kill`` partition logs and does not enter the loss.
    """

    kind: Literal["ppo_kl_fixed"] = "ppo_kl_fixed"
    """Discriminator selecting the fixed-beta PPO-KL objective."""

    beta: Annotated[float, Field(gt=0.0)] = 1.0
    """Fixed scalar multiplier applied to the KL penalty."""

    kl_penalty: Literal["full", "sample", "k3"] = "full"
    """Per-sample KL estimator used in the penalty term."""

    clip_epsilon_for_diagnostics: Annotated[float, Field(gt=0.0)] = 0.2
    """Clip radius used only for partition diagnostics."""
