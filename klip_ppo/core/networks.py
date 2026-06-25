"""Actor-critic MLP for discrete and continuous action spaces."""

from __future__ import annotations

import math
from typing import Literal, cast

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical, Distribution

from klip_ppo.configs.network import MLPConfig
from klip_ppo.core.distributions import (
    PolicyDistParams,
    categorical,
    diag_gaussian,
)

ActionMode = Literal["discrete", "continuous"]


def _activation(name: str) -> type[nn.Module]:
    return {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }[name]


def _ortho_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def _build_mlp(
    in_dim: int, hidden: tuple[int, ...], activation: str, ortho_init: bool
) -> nn.Sequential:
    act_cls = _activation(activation)
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    block = nn.Sequential(*layers)
    if ortho_init:
        block.apply(_ortho_init)
    return block


def _final_linear(
    in_dim: int, out_dim: int, *, ortho_init: bool, gain: float
) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    if ortho_init:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """
    Separate-trunk actor-critic over flat observations.

    Discrete envs produce a ``Categorical``; ``Box`` envs produce a diagonal-Gaussian
    ``Independent(Normal, 1)``. The two trunks are kept independent
    (``share_backbone=False`` by default) so that the value head's gradient does not
    affect the policy's representation and vice versa.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete | gym.spaces.Box,
        config: MLPConfig,
    ) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Box):
            raise NotImplementedError(
                "only flat Box observations are supported "
                f"(got {type(observation_space).__name__})"
            )
        in_dim = int(observation_space.shape[0])
        hidden = tuple(config.hidden_sizes)

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_mode: ActionMode = "discrete"
            self._action_dim = int(action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            self.action_mode = "continuous"
            self._action_dim = int(action_space.shape[0])
        else:
            raise NotImplementedError(
                f"unsupported action space: {type(action_space).__name__}"
            )

        self.policy_trunk = _build_mlp(
            in_dim, hidden, config.activation, config.ortho_init
        )
        self.value_trunk = _build_mlp(
            in_dim, hidden, config.activation, config.ortho_init
        )
        policy_in = hidden[-1] if hidden else in_dim
        value_in = hidden[-1] if hidden else in_dim

        self.policy_head = _final_linear(
            policy_in,
            self._action_dim,
            ortho_init=config.ortho_init,
            gain=0.01,
        )
        self.value_head = _final_linear(
            value_in, 1, ortho_init=config.ortho_init, gain=1.0
        )

        if self.action_mode == "continuous":
            self.log_std = nn.Parameter(
                torch.full((self._action_dim,), float(config.log_std_init))
            )
        else:
            self.register_parameter("log_std", None)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def actor_parameters(self) -> list[nn.Parameter]:
        """
        Parameters that receive gradient from the policy loss (and entropy bonus).

        Includes the policy trunk, policy head, and ``log_std`` for diag-Gaussian
        policies. Because the actor and critic have separate trunks, these parameters do
        not receive gradient from the value loss; that property is what makes
        ``optim/policy_grad_norm/*`` a faithful policy-only norm.
        """
        params: list[nn.Parameter] = list(self.policy_trunk.parameters())
        params.extend(self.policy_head.parameters())
        if self.log_std is not None:
            params.append(self.log_std)
        return params

    def critic_parameters(self) -> list[nn.Parameter]:
        """Parameters that receive gradient only from the value loss."""
        params: list[nn.Parameter] = list(self.value_trunk.parameters())
        params.extend(self.value_head.parameters())
        return params

    def policy_dist_params(self, obs: torch.Tensor) -> PolicyDistParams:
        """
        Return the raw distribution parameters of π(·|obs).

        For categorical policies this is the logits tensor. For diagonal- Gaussian
        policies this is ``(mean, log_std)`` with ``log_std`` broadcast to the batch
        shape. Tensors keep their gradient path to the model parameters; callers under
        ``torch.no_grad()`` will receive non-grad tensors automatically.
        """
        features = self.policy_trunk(obs)
        head_out = self.policy_head(features)
        if self.action_mode == "discrete":
            return PolicyDistParams(kind="categorical", logits=head_out)
        assert self.log_std is not None
        log_std = self.log_std.expand_as(head_out)
        return PolicyDistParams(kind="diag_gaussian", mean=head_out, log_std=log_std)

    def _distribution_from_params(self, params: PolicyDistParams) -> Distribution:
        if params.kind == "categorical":
            assert params.logits is not None
            return categorical(params.logits)
        assert params.mean is not None and params.log_std is not None
        return diag_gaussian(params.mean, params.log_std)

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        return self._distribution_from_params(self.policy_dist_params(obs))

    def _value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.value_trunk(obs)).squeeze(-1)

    def forward(self, obs: torch.Tensor) -> tuple[Distribution, torch.Tensor]:
        return self._distribution(obs), self._value(obs)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return ``V(obs)`` without sampling an action.

        Used by the collector's value-only bootstrap paths (truncated final-state and
        rollout-final-state value estimation). Going through ``act`` for these consumes
        the policy's RNG to sample an action that is then discarded, which complicates
        reproducibility reasoning.
        """
        return self._value(obs)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PolicyDistParams]:
        """
        Sample an action and return ``(action, logprob, value, dist_params)``.

        ``dist_params`` are the raw policy parameters used to build the
        sampling distribution; the collector snapshots them onto the
        rollout buffer so that the training step can later compute
        ``KL(π_old || π_new)`` against the post-update policy.
        """
        params = self.policy_dist_params(obs)
        dist = self._distribution_from_params(params)
        if deterministic:
            if self.action_mode == "discrete":
                action = cast(Categorical, dist).probs.argmax(dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        value = self._value(obs)
        return action, logprob, value, params

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PolicyDistParams]:
        """
        Return ``(logprob, entropy, value, dist_params)`` for stored actions.

        ``dist_params`` are differentiable in θ and reused by the loss strategies to
        compute the full ``KL(π_old || π_new)``.
        """
        params = self.policy_dist_params(obs)
        dist = self._distribution_from_params(params)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self._value(obs)
        return logprob, entropy, value, params
