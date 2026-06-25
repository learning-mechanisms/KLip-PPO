"""Shared PPO hyperparameters across all variants."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from klip_ppo.configs.base import BaseConfig
from klip_ppo.configs.network import OptimiserConfig

AdvantageNormalization = Literal["none", "rollout", "minibatch"]


class PPOAlgoConfigBase(BaseConfig):
    """
    Hyperparameters shared by every PPO variant.

    Variant-specific knobs (``clip_epsilon``, ``beta``, ``kl_target``, etc.) live on the
    subclasses.

    Advantage normalisation modes (``advantage_normalization``):
      - ``"none"``: leave advantages as returned by GAE.
      - ``"rollout"``: subtract mean / divide by std once per rollout,
        before the inner-loop minibatching. This is the convention in
        the IsaacGym / CleanRL PPO baselines.
      - ``"minibatch"``: normalise each minibatch independently. This
        is the SB3 / scratch-code convention.

    The mode does not affect the per-sample β derivation; the
    normalisation is applied uniformly to the advantages that all
    strategies see.

    ``target_kl_stop`` is an early-stop threshold on the inner-loop
    KL estimator (k3, ``approx_kl``). It does not affect the optimised
    loss; it just breaks out of the epoch loop early when the policy
    has moved too far. The k3 comparator is used regardless of
    ``kl_penalty`` (matches the SB3 and CleanRL convention). For
    fixed / adaptive variants whose loss penalises ``kl_full_mean``,
    a threshold transferred from a clip-variant or SB3 reference is
    implicitly calibrated against k3, not against the loss's KL term;
    treat the threshold as a heuristic stopping signal rather than a
    direct bound on the optimised KL.
    """

    optimiser: OptimiserConfig = Field(default_factory=OptimiserConfig)
    """Adam optimiser hyperparameters for policy and value updates."""

    epochs: Annotated[int, Field(gt=0)] = 10
    """Number of inner optimisation passes over each rollout batch."""

    minibatch_size: Annotated[int, Field(gt=0)] = 64
    """Number of rollout samples per PPO minibatch update."""

    gamma: Annotated[float, Field(ge=0.0, le=1.0)] = 0.99
    """Discount factor for future rewards."""

    gae_lambda: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95
    """Trace-decay parameter for generalized advantage estimation."""

    vf_coef: Annotated[float, Field(ge=0.0)] = 0.5
    """Multiplier for the value-function loss."""

    ent_coef: Annotated[float, Field(ge=0.0)] = 0.0
    """Multiplier for the policy entropy bonus."""

    max_grad_norm: Annotated[float, Field(ge=0.0)] = 0.5
    """Global gradient-norm clipping threshold."""

    target_kl_stop: Annotated[float, Field(gt=0.0)] | None = None
    """Optional approximate-KL threshold for early stopping inner epochs."""

    advantage_normalization: AdvantageNormalization = "rollout"
    """Where advantages are normalized before optimisation."""

    clip_value_loss: bool = True
    """Whether to clip value-function updates like PPO-Clip."""

    value_clip_epsilon: Annotated[float, Field(gt=0.0)] = 0.2
    """Radius used when clipping value-function updates."""
