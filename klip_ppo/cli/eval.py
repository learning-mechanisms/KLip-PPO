"""``klip eval`` — evaluate a saved checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
from rich import print as rprint

from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.core.checkpoint import CheckpointManager
from klip_ppo.core.evaluation import evaluate_policy
from klip_ppo.core.networks import ActorCritic
from klip_ppo.envs.gym_env import make_env


def eval_command(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    episodes: int = typer.Option(20, "--episodes"),
    deterministic: bool = typer.Option(True, "--deterministic / --stochastic"),
    seed: int = typer.Option(0, "--seed"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
) -> None:
    """Roll out a saved policy for ``episodes`` episodes; print summary stats."""
    snap_path = run_dir / "snapshot.json"
    if not snap_path.exists():
        raise typer.BadParameter(f"snapshot.json missing in {run_dir}")
    cfg = ExperimentConfig.model_validate(json.loads(snap_path.read_text()))

    env = make_env(cfg.env, seed=seed, env_idx=0)()
    model = ActorCritic(env.observation_space, env.action_space, cfg.network)  # type: ignore[arg-type]

    ckpt_name = checkpoint or "final.pt"
    ckpt_path = run_dir / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise typer.BadParameter(f"checkpoint {ckpt_path} missing")
    state = CheckpointManager(run_dir).load(ckpt_path)
    model.load_state_dict(state["model"])
    env.close()
    collector_state = state.get("collector") or {}
    try:
        eval_stats = evaluate_policy(
            cfg,
            model,
            episodes=episodes,
            deterministic=deterministic,
            seed=seed,
            device=torch.device("cpu"),
            normalizer_state=collector_state.get("normalizer"),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    rprint(
        f"[green]eval[/green] "
        f"episodes={episodes} "
        f"return_mean={eval_stats.mean_return():.3f} "
        f"return_std={eval_stats.std_return():.3f} "
        f"len_mean={eval_stats.mean_length():.1f}"
    )
