"""Torch device + determinism helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn


def pick_device(
    preference: Literal["auto", "cpu", "cuda", "mps"] = "auto",
) -> torch.device:
    """
    Resolve a configured device preference to a concrete ``torch.device``.

    ``auto`` prefers CUDA when available, then Apple MPS, then CPU. Explicit ``cuda`` /
    ``mps`` raise if the requested backend is unavailable so that misconfigured sweeps
    fail loudly instead of silently falling back to CPU.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return torch.device("cuda:0")
    if preference == "mps":
        if not _mps_available():
            raise RuntimeError("device='mps' requested but MPS is not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    is_available = getattr(backend, "is_available", None)
    return bool(is_available() if callable(is_available) else False)


def enable_tf32() -> None:
    """Enable TF32 matmul on supported NVIDIA hardware."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def deterministic_mode(strict: bool) -> None:
    """Toggle torch's deterministic algorithms and cuDNN flags."""
    torch.use_deterministic_algorithms(strict, warn_only=not strict)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def clip_grad_norm(parameters: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor:
    """Thin wrapper around ``torch.nn.utils.clip_grad_norm_`` that always returns a
    scalar tensor."""
    return torch.nn.utils.clip_grad_norm_(list(parameters), max_norm=max_norm)
