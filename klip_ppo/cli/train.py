"""``klip train`` subcommand."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from klip_ppo.cli._common import (
    load_experiment_from_snapshot,
    load_experiment_from_yaml,
)
from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.logging_cfg import WandbMode
from klip_ppo.configs.runtime import ModalGpu
from klip_ppo.runtime.local import LocalRuntime
from klip_ppo.runtime.modal_runtime import ModalRuntime
from klip_ppo.utils.wandb_env import with_wandb_from_env
from klip_ppo.utils.wandb_identity import source_wandb_identity


def train(
    config: Path | None = typer.Argument(
        None,
        exists=False,
        readable=True,
        help="Path to a YAML preset. Mutually exclusive with --from-snapshot.",
    ),
    from_snapshot: Path | None = typer.Option(
        None,
        "--from-snapshot",
        help="Frozen snapshot JSON to replay.",
        readable=True,
    ),
    seed: int | None = typer.Option(None, "--seed", help="Override seed."),
    name: str | None = typer.Option(None, "--name", help="Override experiment name."),
    set_overrides: list[str] = typer.Option(
        [], "--set", help="Dotted override, e.g. algorithm.clip_epsilon=0.1"
    ),
    runtime: str = typer.Option(
        "local", "--runtime", help="Execution backend: 'local' or 'modal'."
    ),
    modal_gpu: ModalGpu | None = typer.Option(
        None,
        "--modal-gpu",
        help="Modal GPU type. Use 'cpu' for CPU-only Modal runs.",
    ),
    allow_dirty_modal: bool = typer.Option(
        False,
        "--allow-dirty-modal",
        help="Allow Modal launch from a dirty git tree and record the diff.",
    ),
    allow_overwrite: bool = typer.Option(
        False, "--allow-overwrite", help="Wipe the run dir if it already exists."
    ),
    wandb_project: str | None = typer.Option(
        None,
        "--wandb-project",
        envvar="WANDB_PROJECT",
        help=(
            "Enable wandb logging with this project. Defaults to $WANDB_PROJECT; "
            "ignored if the config already specifies wandb."
        ),
    ),
    wandb_entity: str | None = typer.Option(
        None,
        "--wandb-entity",
        envvar="WANDB_ENTITY",
        help="Wandb entity. Defaults to $WANDB_ENTITY.",
    ),
    wandb_mode: WandbMode = typer.Option(
        "online",
        "--wandb-mode",
        envvar="WANDB_MODE",
        help="Wandb mode when enabled via --wandb-project.",
    ),
    skip_if_complete: bool = typer.Option(
        False,
        "--skip-if-complete",
        help=(
            "Before running, check WandB for a finished run at this "
            "(group, seed) that reached trainer.total_steps; exit 'skipped' "
            "if found. Requires wandb to be configured."
        ),
    ),
) -> None:
    """Train one PPO Job (one config, one seed, one process, one device)."""
    if runtime not in {"local", "modal"}:
        raise typer.BadParameter("runtime must be 'local' or 'modal'")
    if (config is None) == (from_snapshot is None):
        raise typer.BadParameter("exactly one of CONFIG or --from-snapshot is required")

    if config is not None:
        cfg = load_experiment_from_yaml(
            config, overrides=set_overrides, seed=seed, name=name
        )
        input_path: Path | None = config
        source_identity = source_wandb_identity(config)
    else:
        assert from_snapshot is not None
        cfg = load_experiment_from_snapshot(
            from_snapshot, overrides=set_overrides, seed=seed, name=name
        )
        input_path = None
        source_identity = source_wandb_identity(from_snapshot)

    cfg = _with_runtime(cfg, runtime=runtime, modal_gpu=modal_gpu)
    cfg = _with_wandb_cli(
        cfg,
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
    )
    runtime_impl = (
        LocalRuntime()
        if runtime == "local"
        else ModalRuntime(allow_dirty=allow_dirty_modal)
    )
    result = runtime_impl.run_training(
        cfg,
        seed=cfg.seed,
        input_yaml_path=input_path,
        allow_overwrite=allow_overwrite,
        source_identity=source_identity,
        skip_if_complete=skip_if_complete,
    )
    rprint(
        f"[green]done[/green] run_dir={result.run_dir} "
        f"iterations={result.iterations} env_steps={result.env_steps} "
        f"final_return={result.final_return} exit={result.exit_status}"
    )


def _with_runtime(
    cfg: ExperimentConfig, *, runtime: str, modal_gpu: ModalGpu | None
) -> ExperimentConfig:
    updates: dict[str, object] = {"backend": runtime}
    if modal_gpu is not None:
        updates["modal_gpu"] = modal_gpu
    return cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update=updates)})


def _with_wandb_cli(
    cfg: ExperimentConfig,
    *,
    project: str | None,
    entity: str | None,
    mode: WandbMode,
) -> ExperimentConfig:
    return with_wandb_from_env(cfg, project=project, entity=entity, mode=mode)
