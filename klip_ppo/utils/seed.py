"""Deterministic seeding across python, numpy, torch, and gymnasium."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedReport:
    """What ``set_seed`` actually configured."""

    seed: int
    python_random: bool
    numpy: bool
    torch_cpu: bool
    torch_cuda: bool


def set_seed(seed: int) -> SeedReport:
    """
    Seed python, numpy, torch CPU + CUDA, and PYTHONHASHSEED.

    Gymnasium envs are seeded at reset time with this seed (or derived offsets) by the
    collector, not here.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_ok = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_ok = True
    except ImportError:
        pass

    torch_cpu_ok = False
    torch_cuda_ok = False
    try:
        import torch

        torch.manual_seed(seed)
        torch_cpu_ok = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch_cuda_ok = True
    except ImportError:
        pass

    return SeedReport(
        seed=seed,
        python_random=True,
        numpy=numpy_ok,
        torch_cpu=torch_cpu_ok,
        torch_cuda=torch_cuda_ok,
    )
