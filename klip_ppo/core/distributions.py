"""
Distribution helpers shared by discrete and continuous policies.

Three layers are exposed:

1. ``PolicyDistParams`` — a small dataclass holding the raw parameters of
   a stochastic policy at a batch of states. Categorical policies use
   ``logits``; diagonal-Gaussian policies use ``(mean, log_std)``. The
   trainer snapshots these at rollout time (detached) so that the
   training step can compute the full ``KL(π_old || π_new)`` against
   the differentiable post-update params.

2. ``diag_gaussian`` / ``categorical`` — builders for the standard
   ``torch.distributions`` types used by the actor-critic.

3. ``kl_old_new`` and friends — per-sample KL divergences between two
   sets of params. Old-side params are detached inside the helper so
   gradients flow only through the new-side params, regardless of how
   the caller built the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn.functional as functional
from torch.distributions import Categorical, Distribution, Independent, Normal

PolicyKind = Literal["categorical", "diag_gaussian"]


@dataclass
class PolicyDistParams:
    """
    Raw parameters of a stochastic policy at a batch of states.

    Invariants:
      - ``kind == "categorical"`` → ``logits`` is set; ``mean``, ``log_std`` are None.
      - ``kind == "diag_gaussian"`` → ``mean`` and ``log_std`` are set; ``logits`` is None.

    Tensors are shape ``(B, ...)``; the batch axis is preserved by
    ``index(idx)`` and ``to(device)``. ``detach()`` returns a new
    snapshot whose tensors no longer track gradients (used when storing
    old-policy params on the rollout buffer).
    """

    kind: PolicyKind
    logits: torch.Tensor | None = None
    mean: torch.Tensor | None = None
    log_std: torch.Tensor | None = None

    def to(self, device: torch.device) -> PolicyDistParams:
        return PolicyDistParams(
            kind=self.kind,
            logits=self.logits.to(device) if self.logits is not None else None,
            mean=self.mean.to(device) if self.mean is not None else None,
            log_std=self.log_std.to(device) if self.log_std is not None else None,
        )

    def index(self, idx: torch.Tensor) -> PolicyDistParams:
        return PolicyDistParams(
            kind=self.kind,
            logits=self.logits[idx] if self.logits is not None else None,
            mean=self.mean[idx] if self.mean is not None else None,
            log_std=self.log_std[idx] if self.log_std is not None else None,
        )

    def detach(self) -> PolicyDistParams:
        return PolicyDistParams(
            kind=self.kind,
            logits=self.logits.detach() if self.logits is not None else None,
            mean=self.mean.detach() if self.mean is not None else None,
            log_std=self.log_std.detach() if self.log_std is not None else None,
        )


class PolicyDistribution(Protocol):
    """
    The subset of ``torch.distributions.Distribution`` we rely on.

    Both ``Categorical`` and ``Independent(Normal, 1)`` satisfy this.
    """

    def sample(self) -> torch.Tensor: ...

    def log_prob(self, value: torch.Tensor) -> torch.Tensor: ...

    def entropy(self) -> torch.Tensor: ...


def diag_gaussian(mean: torch.Tensor, log_std: torch.Tensor) -> Distribution:
    std = log_std.exp().expand_as(mean)
    return Independent(Normal(mean, std), 1)


def categorical(logits: torch.Tensor) -> Distribution:
    return Categorical(logits=logits)


def gaussian_kl_old_new(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample ``KL(N(old_mean, exp(old_log_std)^2) || N(new_mean,
    exp(new_log_std)^2))``.

    Diagonal-Gaussian closed form, summed over the action dimension. Old-side inputs are
    detached inside the function so the result is differentiable only with respect to
    ``new_mean`` and ``new_log_std``.

    Returned tensor has shape ``(B,)`` when inputs are ``(B, action_dim)``.
    """
    old_mean = old_mean.detach()
    old_log_std = old_log_std.detach()
    old_var = (2.0 * old_log_std).exp()
    new_var = (2.0 * new_log_std).exp()
    per_dim = (
        new_log_std
        - old_log_std
        + 0.5 * (old_var + (old_mean - new_mean).pow(2)) / new_var
        - 0.5
    )
    return per_dim.sum(dim=-1)


def categorical_kl_old_new(
    old_logits: torch.Tensor, new_logits: torch.Tensor
) -> torch.Tensor:
    """
    Per-sample ``KL(softmax(old_logits) || softmax(new_logits))``.

    Closed-form discrete KL. Old logits are detached inside the function. Returned
    tensor has shape ``(B,)``.
    """
    old_logits = old_logits.detach()
    old_logp = functional.log_softmax(old_logits, dim=-1)
    new_logp = functional.log_softmax(new_logits, dim=-1)
    old_p = old_logp.exp()
    return (old_p * (old_logp - new_logp)).sum(dim=-1)


def kl_old_new(old: PolicyDistParams, new: PolicyDistParams) -> torch.Tensor:
    """
    Per-sample ``KL(π_old || π_new)`` for matching distribution kinds.

    Gradients flow only into ``new``'s tensors; the helpers detach the old side
    internally.
    """
    if old.kind != new.kind:
        raise ValueError(
            f"kind mismatch in kl_old_new: old={old.kind!r}, new={new.kind!r}"
        )
    if old.kind == "categorical":
        assert old.logits is not None and new.logits is not None
        return categorical_kl_old_new(old.logits, new.logits)
    assert old.mean is not None and new.mean is not None
    assert old.log_std is not None and new.log_std is not None
    return gaussian_kl_old_new(old.mean, old.log_std, new.mean, new.log_std)


def sampled_forward_kl(
    old_logprob: torch.Tensor, new_logprob: torch.Tensor
) -> torch.Tensor:
    """Per-sample value of the single-sample forward-KL estimator.

    ``KL(π_old || π_new) = E_{a ~ π_old}[log π_old(a|s) - log π_new(a|s)]``
    so the sample-level value (with action ``a`` drawn from π_old at
    rollout time) is ``log π_old(a_t|s_t) - log π_new(a_t|s_t)``.
    Differentiable in θ via ``new_logprob``.
    """
    return old_logprob - new_logprob
