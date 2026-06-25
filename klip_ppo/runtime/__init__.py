"""Execution backends and sweep orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klip_ppo.runtime.base import RunResult, Runtime
    from klip_ppo.runtime.local import LocalRuntime, worker_main
    from klip_ppo.runtime.sweep import SweepResult, SweepRunner

__all__ = [
    "LocalRuntime",
    "RunResult",
    "Runtime",
    "SweepResult",
    "SweepRunner",
    "worker_main",
]


def __getattr__(name: str) -> object:
    """Load runtime implementations only when requested."""
    if name in {"RunResult", "Runtime"}:
        from klip_ppo.runtime.base import RunResult, Runtime

        return {"RunResult": RunResult, "Runtime": Runtime}[name]
    if name in {"LocalRuntime", "worker_main"}:
        from klip_ppo.runtime.local import LocalRuntime, worker_main

        return {"LocalRuntime": LocalRuntime, "worker_main": worker_main}[name]
    if name in {"SweepResult", "SweepRunner"}:
        from klip_ppo.runtime.sweep import SweepResult, SweepRunner

        return {"SweepResult": SweepResult, "SweepRunner": SweepRunner}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
