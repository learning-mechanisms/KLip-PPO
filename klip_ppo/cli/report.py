"""``klip report build`` — assemble a markdown report from artifacts."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich import print as rprint

from klip_ppo.configs.logging_cfg import WandbMode
from klip_ppo.utils.paths import REPORTS_DIR, RUNS_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("build")
def build(
    out: Path | None = typer.Option(None, "--out"),
    runs_root: Path = typer.Option(RUNS_DIR, "--runs-root"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity"),
    wandb_run_name: str | None = typer.Option(None, "--wandb-run-name"),
    wandb_mode: WandbMode = typer.Option("online", "--wandb-mode"),
    wandb_alias: list[str] = typer.Option(["latest"], "--wandb-alias"),
) -> None:
    from klip_ppo.research.report import build_markdown_report

    target = out or (REPORTS_DIR / date.today().isoformat() / "report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    build_markdown_report(runs_root, target)
    rprint(f"[green]wrote[/green] {target}")
    if wandb_project is not None:
        if wandb_mode == "disabled":
            rprint("[yellow]wandb disabled; skipped upload[/yellow]")
            return
        from klip_ppo.research.aggregation import load_runs
        from klip_ppo.research.tables import final_returns, partition_stats
        from klip_ppo.utils.wandb_utils import publish_report_artifact

        df = load_runs(runs_root)
        publish_report_artifact(
            target,
            project=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name or f"report__{target.parent.name}",
            mode=wandb_mode,
            aliases=wandb_alias,
            final_returns=final_returns(df),
            partition_stats=partition_stats(df),
            metadata={"runs_root": str(runs_root), "report_path": str(target)},
        )
        rprint(f"[green]uploaded to wandb[/green] {wandb_project}")
