"""
Shared scaffold for PPO loss strategies.

The trainer treats every variant through the ``Strategy`` Protocol. Concrete subclasses
fill in ``_policy_loss``; everything else (entropy bonus, value loss, KL diagnostics,
partition stats) lives here so the four variants stay small and inspectable.

Two lifecycle hooks are exposed:

- ``on_epoch_end(epoch_agg)``: called after each inner epoch with the
  per-epoch aggregate. Default is a no-op. Variants can override for
  per-epoch diagnostics, but the PPO paper's adaptive β rule mutates
  β only after the rollout update, not after each epoch — see
  ``on_rollout_end``.
- ``on_rollout_end(rollout_agg, *, final_kl=None)``: called once per
  rollout update with the aggregate over all minibatches and epochs,
  plus a per-estimator dict of post-update KL means computed in one
  no-grad pass over the rollout under the final policy. This is where
  ``KLAdaptiveStrategy`` updates β; ``final_kl`` is the controller
  variable, matching Schulman 2017 §2.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

import torch

from klip_ppo.configs.algorithm.base import PPOAlgoConfigBase
from klip_ppo.core.distributions import PolicyDistParams, kl_old_new, sampled_forward_kl
from klip_ppo.core.losses import (
    ClipPartition,
    approx_kl_from_logratio,
    clipped_value_loss,
    partition_indices,
    policy_ratio,
)
from klip_ppo.core.networks import ActorCritic
from klip_ppo.core.rollout import PPOMinibatch

KLPenaltyKind = Literal["full", "sample", "k3"]
_MIN_REDUCE_KEYS: frozenset[str] = frozenset({"ratio_min"})
_MAX_REDUCE_KEYS: frozenset[str] = frozenset({"ratio_max"})

# Encoding for per-sample hard-partition labels. Matches the I_in / I_pass /
# I_kill triple plus the "unclassified" residual (advantage exactly zero with
# ratio out of band). The trainer uses these to compute partition migration
# rate across inner epochs in ``diagnostic_mode = "full"``.
PARTITION_LABEL_IN: int = 0
PARTITION_LABEL_PASS: int = 1
PARTITION_LABEL_KILL: int = 2
PARTITION_LABEL_UNCLASSIFIED: int = 3


@dataclass
class StrategyOutputs:
    """What every Strategy.step returns to the trainer."""

    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    total_loss: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    sample_diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
    # Per-sample hard-partition labels under the current policy (int8 in
    # [0, 3]). Always populated by ``BasePPOLossStrategy.step``. Consumed by
    # ``diagnostic_mode = "full"`` to track per-sample partition migration.
    per_sample_labels: torch.Tensor | None = None


@dataclass
class EpochAggregate:
    """
    Sum / mean accumulator used to summarise a single epoch or a full rollout.

    The same dataclass is used for the per-epoch aggregate (handed to ``on_epoch_end``)
    and for the rollout-wide aggregate (handed to ``on_rollout_end``). ``counts`` is the
    number of minibatches fed in.
    """

    counts: int = 0
    sums: dict[str, float] = field(default_factory=dict)
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)

    def update(
        self,
        out: StrategyOutputs,
        *,
        policy_grad_norm: float | None = None,
        global_grad_norm: float | None = None,
    ) -> None:
        """
        Fold one minibatch's outputs into the running aggregate.

        ``policy_grad_norm`` is the pre-clip gradient norm restricted to the actor
        parameters (policy trunk + head + ``log_std``). ``global_grad_norm`` is the pre-
        clip norm over the whole model (actor + critic), which is what
        ``torch.nn.utils.clip_grad_norm_`` returns. Keeping them separate lets the soft-
        clipping analysis isolate policy-gradient variance from value-loss effects (see
        paper §6 future-work directions).
        """
        self.counts += 1
        self._add("policy_loss", float(out.policy_loss.detach()))
        self._add("value_loss", float(out.value_loss.detach()))
        self._add("entropy", float(out.entropy.detach()))
        self._add("total_loss", float(out.total_loss.detach()))
        for k, v in out.diagnostics.items():
            self._add(k, float(v.detach()))
        if policy_grad_norm is not None:
            self._add("policy_grad_norm", policy_grad_norm)
            self._add("policy_grad_norm_sq", policy_grad_norm * policy_grad_norm)
        if global_grad_norm is not None:
            self._add("global_grad_norm", global_grad_norm)
            self._add("global_grad_norm_sq", global_grad_norm * global_grad_norm)

    def _add(self, key: str, value: float) -> None:
        if key in _MIN_REDUCE_KEYS:
            self.mins[key] = min(value, self.mins.get(key, value))
            return
        if key in _MAX_REDUCE_KEYS:
            self.maxs[key] = max(value, self.maxs.get(key, value))
            return
        self.sums[key] = self.sums.get(key, 0.0) + value

    def mean(self, key: str) -> float | None:
        if key in self.mins:
            return self.mins[key]
        if key in self.maxs:
            return self.maxs[key]
        if self.counts == 0 or key not in self.sums:
            return None
        return self.sums[key] / self.counts

    def as_dict(self) -> dict[str, float]:
        if self.counts == 0:
            return {}
        out = {k: v / self.counts for k, v in self.sums.items()}
        out.update(self.mins)
        out.update(self.maxs)
        return out


class Strategy(Protocol):
    name: str

    def step(self, mb: PPOMinibatch, model: ActorCritic) -> StrategyOutputs: ...

    def on_epoch_end(self, agg: EpochAggregate) -> None: ...

    def on_rollout_end(
        self,
        agg: EpochAggregate,
        *,
        final_kl: dict[str, float] | None = None,
    ) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass
class _Shared:
    """
    Quantities computed once per minibatch and reused by every variant.

    KL estimators are returned as per-sample ``(B,)`` tensors so variants
    can select among them (see ``select_kl_penalty``). All three KL
    streams are differentiable in θ; the ``full`` stream additionally
    requires that ``mb.old_dist_params`` was correctly snapshotted at
    rollout time (which the standard collector guarantees).

    ``full_kl_t``  : closed-form ``KL(π_old(·|s_t) || π_new(·|s_t))``.
    ``sample_kl_t``: single-sample ``log π_old(a_t|s_t) - log π_new(a_t|s_t)``.
    ``k3_kl_t``    : Schulman's k3 ``(w_t - 1) - log w_t``.

    ``partition`` uses the *detached* ratio; gradients never flow
    through the partition masks.
    """

    new_logprobs: torch.Tensor
    new_dist_params: PolicyDistParams
    entropy: torch.Tensor
    values: torch.Tensor
    log_ratio: torch.Tensor
    ratio: torch.Tensor
    approx_kl: torch.Tensor
    value_loss: torch.Tensor
    partition: ClipPartition
    full_kl_t: torch.Tensor
    sample_kl_t: torch.Tensor
    k3_kl_t: torch.Tensor


def select_kl_penalty(shared: _Shared, kind: KLPenaltyKind) -> torch.Tensor:
    """Pick the per-sample KL estimator named by ``kind``."""
    if kind == "full":
        return shared.full_kl_t
    if kind == "sample":
        return shared.sample_kl_t
    if kind == "k3":
        return shared.k3_kl_t
    raise ValueError(f"unknown kl_penalty kind: {kind!r}")


class BasePPOLossStrategy:
    """
    Common machinery for every PPO variant.

    Variants override ``_policy_loss`` (returns the policy loss tensor
    plus a dict of variant-specific diagnostics). Total loss is then
    composed as ``policy + vf_coef * value - ent_coef * entropy``.
    """

    name: str = "ppo_base"

    def __init__(self, cfg: PPOAlgoConfigBase) -> None:
        self.cfg = cfg

    def _shared(self, mb: PPOMinibatch, model: ActorCritic) -> _Shared:
        new_logprobs, entropy, values, new_params = model.evaluate_actions(
            mb.obs, mb.actions
        )
        log_ratio = new_logprobs - mb.old_logprobs
        ratio = policy_ratio(new_logprobs, mb.old_logprobs)
        approx_kl = approx_kl_from_logratio(log_ratio.detach())
        v_loss = clipped_value_loss(
            values,
            mb.old_values,
            mb.returns,
            self.cfg.value_clip_epsilon,
            clip=self.cfg.clip_value_loss,
        )
        partition = partition_indices(
            ratio.detach(), mb.advantages, self._diagnostics_epsilon()
        )
        full_kl_t = kl_old_new(mb.old_dist_params, new_params)
        sample_kl_t = sampled_forward_kl(mb.old_logprobs, new_logprobs)
        k3_kl_t = (ratio - 1.0) - log_ratio
        return _Shared(
            new_logprobs=new_logprobs,
            new_dist_params=new_params,
            entropy=entropy,
            values=values,
            log_ratio=log_ratio,
            ratio=ratio,
            approx_kl=approx_kl,
            value_loss=v_loss,
            partition=partition,
            full_kl_t=full_kl_t,
            sample_kl_t=sample_kl_t,
            k3_kl_t=k3_kl_t,
        )

    def _diagnostics_epsilon(self) -> float:
        """``clip_epsilon`` if the variant has one; else a fallback for diagnostics."""
        eps = getattr(self.cfg, "clip_epsilon", None)
        if eps is None:
            eps = getattr(self.cfg, "clip_epsilon_for_diagnostics", 0.2)
        return float(cast(float, eps))

    def step(self, mb: PPOMinibatch, model: ActorCritic) -> StrategyOutputs:
        shared = self._shared(mb, model)
        policy_loss, extras = self._policy_loss(mb, shared)
        entropy_mean = shared.entropy.mean()
        total = (
            policy_loss
            + self.cfg.vf_coef * shared.value_loss
            - self.cfg.ent_coef * entropy_mean
        )
        eps = self._diagnostics_epsilon()
        ratio_det = shared.ratio.detach()
        # Standard PPO definition of clip_fraction: fraction of samples whose
        # ratio falls outside [1-eps, 1+eps], independent of advantage sign.
        clip_fraction = ((ratio_det - 1.0).abs() > eps).float().mean()
        in_band = shared.partition.in_band.float().mean()
        in_pass = shared.partition.pass_through.float().mean()
        in_kill = shared.partition.kill.float().mean()
        # Out-of-band but with zero advantage are not assigned to any of
        # the three named partitions; track them so the four numbers sum
        # to 1.
        unclassified = (
            1.0
            - shared.partition.in_band.float()
            - shared.partition.pass_through.float()
            - shared.partition.kill.float()
        ).mean()
        diag: dict[str, torch.Tensor] = {
            "approx_kl": shared.approx_kl,
            "ratio_mean": ratio_det.mean(),
            "ratio_min": ratio_det.min(),
            "ratio_max": ratio_det.max(),
            "ratio_p05": torch.quantile(ratio_det, 0.05),
            "ratio_p95": torch.quantile(ratio_det, 0.95),
            "clip_fraction": clip_fraction,
            "frac_in_I_in": in_band,
            "frac_in_I_pass": in_pass,
            "frac_in_I_kill": in_kill,
            "frac_in_I_unclassified": unclassified,
            "kl_full_mean": shared.full_kl_t.detach().mean(),
            "kl_sample_mean": shared.sample_kl_t.detach().mean(),
        }
        diag.update(extras)
        sample_diag = {
            k: v.detach() for k, v in self._sample_diagnostics(mb, shared).items()
        }
        return StrategyOutputs(
            policy_loss=policy_loss.detach(),
            value_loss=shared.value_loss.detach(),
            entropy=entropy_mean.detach(),
            total_loss=total,
            diagnostics=diag,
            sample_diagnostics=sample_diag,
            per_sample_labels=_encode_partition_labels(shared.partition),
        )

    def _policy_loss(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def _sample_diagnostics(
        self, mb: PPOMinibatch, shared: _Shared
    ) -> dict[str, torch.Tensor]:
        return {}

    def on_epoch_end(self, agg: EpochAggregate) -> None:  # default: no-op
        return None

    def on_rollout_end(
        self,
        agg: EpochAggregate,
        *,
        final_kl: dict[str, float] | None = None,
    ) -> None:
        """
        No-op default; ``final_kl`` is only consumed by the adaptive variant.

        ``final_kl`` is a dict keyed by KL-penalty kind (``"full"``, ``"sample"``,
        ``"k3"``) of mean KL between the rollout policy and the post-update policy,
        recomputed in one no-grad pass over the full rollout. The adaptive controller
        uses it instead of the during-training aggregate.
        """
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


def _encode_partition_labels(partition: ClipPartition) -> torch.Tensor:
    """Encode the I_in / I_pass / I_kill / unclassified triple as an int8 tensor."""
    labels = torch.full_like(
        partition.in_band, PARTITION_LABEL_UNCLASSIFIED, dtype=torch.int8
    )
    labels[partition.in_band] = PARTITION_LABEL_IN
    labels[partition.pass_through] = PARTITION_LABEL_PASS
    labels[partition.kill] = PARTITION_LABEL_KILL
    return labels.detach()
