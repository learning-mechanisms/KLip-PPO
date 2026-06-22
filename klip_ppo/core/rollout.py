"""Rollout buffer + collector for on-policy PPO."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from klip_ppo.core.distributions import PolicyDistParams
from klip_ppo.core.networks import ActorCritic


@dataclass
class RolloutBatch:
    """
    A fixed-size on-policy rollout, flattened to ``(T*E, ...)`` views.

    Tensors are stored on the trainer's device. ``obs`` and ``actions`` keep their
    original action shape; everything else is 1-D over the flattened ``T*E`` batch.
    ``old_dist_params`` snapshots the policy parameters used to sample each (s, a)
    transition; it powers the full ``KL(π_old || π_new)`` computed by the loss
    strategies.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_dist_params: PolicyDistParams

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def iter_minibatches(
        self, minibatch_size: int, generator: torch.Generator | None = None
    ) -> Iterator[PPOMinibatch]:
        for mb, _ in self.iter_minibatches_with_indices(minibatch_size, generator):
            yield mb

    def iter_minibatches_with_indices(
        self, minibatch_size: int, generator: torch.Generator | None = None
    ) -> Iterator[tuple[PPOMinibatch, torch.Tensor]]:
        """
        Same as ``iter_minibatches`` but also yields the rollout-position index tensor.

        The index tensor lets diagnostic code align per-sample quantities (e.g. hard
        partition labels) across inner epochs even after the minibatch permutation has
        been reshuffled. Used by ``diagnostic_mode = "full"`` to compute partition
        migration rate; not needed for the loss itself.
        """
        n = len(self)
        if minibatch_size > n:
            minibatch_size = n
        perm = torch.randperm(n, generator=generator, device=self.obs.device)
        for start in range(0, n, minibatch_size):
            idx = perm[start : start + minibatch_size]
            mb = PPOMinibatch(
                obs=self.obs[idx],
                actions=self.actions[idx],
                old_logprobs=self.logprobs[idx],
                old_values=self.values[idx],
                advantages=self.advantages[idx],
                returns=self.returns[idx],
                old_dist_params=self.old_dist_params.index(idx),
            )
            yield mb, idx


@dataclass
class PPOMinibatch:
    """A single minibatch handed to a Strategy."""

    obs: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_dist_params: PolicyDistParams


class Collector(Protocol):
    """Anything that turns a policy into a fresh ``RolloutBatch``."""

    @property
    def num_envs(self) -> int: ...

    @property
    def n_steps(self) -> int: ...

    def collect(self, policy: ActorCritic) -> tuple[RolloutBatch, EpisodeStats]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


@dataclass
class EpisodeStats:
    """
    Recent-episode return / length statistics from this rollout.

    Two return streams are tracked separately:

    - ``raw_returns``: episode returns measured *before* any reward
      normalisation or scaling wrappers. Comes from a
      ``RecordEpisodeStatistics`` wrapper inserted near the base env;
      these are the numbers that compare apples-to-apples with the
      published PPO results on the same benchmark.
    - ``wrapped_returns``: episode returns of the rewards the trainer
      actually observed (post collector-owned normalisation / scaling).
      Useful for diagnostics when normalisation is on.

    ``mean_return()`` prefers ``raw_returns`` when available and falls
    back to ``wrapped_returns`` so callers see literature-comparable
    numbers by default.
    """

    raw_returns: list[float]
    wrapped_returns: list[float]
    lengths: list[float]

    @property
    def n(self) -> int:
        return len(self.wrapped_returns)

    def mean_raw_return(self) -> float | None:
        return float(np.mean(self.raw_returns)) if self.raw_returns else None

    def mean_wrapped_return(self) -> float | None:
        return float(np.mean(self.wrapped_returns)) if self.wrapped_returns else None

    def mean_return(self) -> float | None:
        raw = self.mean_raw_return()
        return raw if raw is not None else self.mean_wrapped_return()

    def iqm_return(self) -> float | None:
        source = self.raw_returns or self.wrapped_returns
        if len(source) < 4:
            return float(np.mean(source)) if source else None
        arr = np.asarray(source)
        lo, hi = np.quantile(arr, [0.25, 0.75])
        mask = (arr >= lo) & (arr <= hi)
        return float(arr[mask].mean()) if mask.any() else float(arr.mean())

    def mean_length(self) -> float | None:
        return float(np.mean(self.lengths)) if self.lengths else None
