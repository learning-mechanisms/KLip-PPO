"""Drop identifying fields from a run lock for blind release."""

from __future__ import annotations

from typing import Any


def blind(lock: dict[str, Any]) -> dict[str, Any]:
    """Keep run structure (algo, env, seed); drop project and run ids."""

    def keep(entry: dict[str, Any]) -> dict[str, Any]:
        return {"algo": entry["algo"], "env": entry["env"], "seed": entry["seed"]}

    return {
        "metrics": lock["metrics"],
        "baselines": [keep(e) for e in lock["baselines"]],
        "sweeps": [keep(e) for e in lock["sweeps"]],
    }
