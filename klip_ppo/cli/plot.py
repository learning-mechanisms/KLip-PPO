"""``klip plot {curves, kl-vs-clip}`` — figure builders."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich import print as rprint

from klip_ppo.configs.logging_cfg import WandbMode
from klip_ppo.utils.paths import REPORTS_DIR, RUNS_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("curves")
def curves(
    algos: list[str] = typer.Option([], "--algos"),
    env: str | None = typer.Option(None, "--env"),
    out: Path | None = typer.Option(None, "--out"),
    runs_root: Path = typer.Option(RUNS_DIR, "--runs-root"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity"),
    wandb_run_name: str | None = typer.Option(None, "--wandb-run-name"),
    wandb_mode: WandbMode = typer.Option("online", "--wandb-mode"),
    wandb_alias: list[str] = typer.Option(["latest"], "--wandb-alias"),
) -> None:
    """Median + IQR learning curves across the matching runs."""
    from klip_ppo.research.aggregation import load_runs
    from klip_ppo.research.plotting import DEFAULT_PLOT_SUFFIX, plot_learning_curves

    df = load_runs(runs_root, algos=algos, env=env)
    if df.empty:
        rprint("[yellow]no runs matched[/yellow]")
        raise typer.Exit(code=1)
    target = out or (
        REPORTS_DIR / date.today().isoformat() / f"learning_curves{DEFAULT_PLOT_SUFFIX}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(df, target)
    rprint(f"[green]wrote[/green] {target}")
    if wandb_project is not None:
        if wandb_mode == "disabled":
            rprint("[yellow]wandb disabled; skipped upload[/yellow]")
            return
        from klip_ppo.utils.wandb_utils import publish_file_artifact

        publish_file_artifact(
            target,
            project=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name or f"plot__curves__{target.parent.name}",
            mode=wandb_mode,
            job_type="plot",
            artifact_type="plot",
            aliases=wandb_alias,
            metadata={
                "kind": "learning_curves",
                "runs_root": str(runs_root),
                "algos": algos,
                "env": env,
            },
            log_image_key="plot/learning_curves",
        )
        rprint(f"[green]uploaded to wandb[/green] {wandb_project}")


@app.command("kl-vs-clip")
def kl_vs_clip(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True),
    out: Path | None = typer.Option(None, "--out"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity"),
    wandb_run_name: str | None = typer.Option(None, "--wandb-run-name"),
    wandb_mode: WandbMode = typer.Option("online", "--wandb-mode"),
    wandb_alias: list[str] = typer.Option(["latest"], "--wandb-alias"),
) -> None:
    """Diagnostic plot for one run: KL, clip-fraction, and I_kill share over time."""
    from klip_ppo.research.plotting import DEFAULT_PLOT_SUFFIX, plot_kl_vs_clip

    target = out or (
        REPORTS_DIR
        / date.today().isoformat()
        / f"{run_dir.name}__kl_vs_clip{DEFAULT_PLOT_SUFFIX}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    plot_kl_vs_clip(run_dir, target)
    rprint(f"[green]wrote[/green] {target}")
    if wandb_project is not None:
        if wandb_mode == "disabled":
            rprint("[yellow]wandb disabled; skipped upload[/yellow]")
            return
        from klip_ppo.utils.wandb_utils import publish_file_artifact

        publish_file_artifact(
            target,
            project=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name or f"plot__kl_vs_clip__{run_dir.name}",
            mode=wandb_mode,
            job_type="plot",
            artifact_type="plot",
            aliases=wandb_alias,
            metadata={"kind": "kl_vs_clip", "run_dir": str(run_dir)},
            log_image_key="plot/kl_vs_clip",
        )
        rprint(f"[green]uploaded to wandb[/green] {wandb_project}")
