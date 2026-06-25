"""Network and optimiser configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from klip_ppo.configs.base import BaseConfig

Activation = Literal["tanh", "relu", "elu", "gelu"]


class OptimiserConfig(BaseConfig):
    """
    Adam-style optimiser hyperparameters.

    ``anneal_lr=True`` linearly decays the learning rate from ``lr`` to ``0`` over the
    training run (CleanRL convention). Default is ``False`` so isolated tests that
    instantiate this config directly keep the constant-LR baseline.
    """

    lr: Annotated[float, Field(gt=0.0)] = 3e-4
    """Initial Adam learning rate before optional annealing."""

    eps: Annotated[float, Field(gt=0.0)] = 1e-5
    """Numerical stability epsilon passed to Adam."""

    beta1: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.9
    """Adam first-moment decay coefficient."""

    beta2: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.999
    """Adam second-moment decay coefficient."""

    weight_decay: Annotated[float, Field(ge=0.0)] = 0.0
    """Weight-decay coefficient applied by the optimiser."""

    anneal_lr: bool = Field(
        default=False,
        description="Whether to linearly decay the learning rate to zero.",
    )


class MLPConfig(BaseConfig):
    """Actor-critic MLP architecture (shared for policy and value heads)."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    """Widths of hidden layers in the actor and critic MLPs."""

    activation: Activation = "tanh"
    """Nonlinearity used after each hidden layer."""

    ortho_init: bool = True
    """Whether to use orthogonal parameter initialization."""

    log_std_init: float = 0.0
    """Initial log standard deviation for Gaussian policy actions."""

    share_backbone: bool = False
    """Whether policy and value networks share the same MLP trunk."""

    @model_validator(mode="after")
    def _supported_backbone(self) -> MLPConfig:
        if self.share_backbone:
            raise ValueError(
                "network.share_backbone=true is not implemented; the current "
                "ActorCritic uses separate policy/value trunks"
            )
        return self


NetworkConfig = MLPConfig
