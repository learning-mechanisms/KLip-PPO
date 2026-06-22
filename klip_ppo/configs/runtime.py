"""Runtime backend configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from klip_ppo.configs.base import BaseConfig

ModalGpu = Literal[
    "cpu",
    "T4",
    "L4",
    "A10",
    "L40S",
    "A100",
    "A100-40GB",
    "A100-80GB",
    "H100",
]

DEFAULT_MODAL_TIMEOUT_SECONDS = 24 * 60 * 60


class RuntimeConfig(BaseConfig):
    """
    Where and how the Job process executes.

    A Job is always single-device; there is no ``gpus`` field. Multi-GPU throughput is
    obtained by launching multiple Jobs via the Sweep runner, each pinned to one device.
    """

    backend: Literal["local", "modal"] = "local"
    """Execution backend that launches the training job."""

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    """Torch device selector used inside the job process."""

    deterministic: bool = False
    """Whether to request deterministic Torch backend behavior."""

    cudnn_benchmark: bool = True
    """Whether cuDNN may autotune kernels when determinism is off."""

    mixed_precision: Literal["off", "fp16", "bf16"] = "off"
    """Mixed-precision mode; only ``off`` is currently implemented."""

    num_threads: int | None = None
    """Optional override for Torch CPU worker threads."""

    modal_gpu: ModalGpu = "L4"
    """Modal GPU class requested for remote jobs."""

    modal_timeout_seconds: Annotated[int, Field(gt=0)] = DEFAULT_MODAL_TIMEOUT_SECONDS
    """Maximum Modal job runtime before timeout."""

    @model_validator(mode="after")
    def _supported_precision(self) -> RuntimeConfig:
        if self.mixed_precision != "off":
            raise ValueError(
                "runtime.mixed_precision is exposed for future work but is not "
                "implemented yet; set it to 'off'"
            )
        return self
