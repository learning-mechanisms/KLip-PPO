"""PPO with soft relaxations of the clipped surrogate."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from klip_ppo.configs.algorithm.base import PPOAlgoConfigBase

SoftClipMethod = Literal["linear_ramp", "sigmoid", "soft_min"]


class PPOSoftClipConfig(PPOAlgoConfigBase):
    """
    PPO with a temperature-like soft clipping objective.

    ``softness`` is the public softness/temperature parameter. Larger values make the
    clip boundary softer; smaller values approach hard PPO-Clip.

    Reader caveat for ``method = "sigmoid"``: the sigmoid gate has a nonzero tail
    inside the trust region. At ``clip_epsilon = 0.2`` and ``softness = 0.05`` it
    already brakes ~12% of the unclipped gradient at ratio 1.1, deep inside I_in.
    ``method = "linear_ramp"`` and ``"soft_min"`` do not have this property at the
    same ``softness``. Watch ``soft_clip/gate/mean/I_in`` and
    ``soft_clip/gate/mean/I_pass`` in the logs: a nonzero value means softening is
    reaching past the kill boundary, and the three methods are not strictly
    comparable at a shared ``softness`` until you equalise the inside-band gate mean.
    """

    kind: Literal["ppo_soft_clip"] = "ppo_soft_clip"
    """Discriminator selecting the soft-clipping PPO objective."""

    clip_epsilon: Annotated[float, Field(gt=0.0)] = 0.2
    """Hard PPO-Clip radius around which soft clipping is centered."""

    method: SoftClipMethod = "linear_ramp"
    """Soft clipping relaxation used at the kill boundary."""

    softness: Annotated[float, Field(gt=0.0)] = 0.05
    """Width or temperature controlling how gradual the boundary is."""
