"""
Top-level Typer entry point.

Exposed via ``klip`` console script.
"""

from __future__ import annotations

import typer

from klip_ppo.cli import (
    artifacts,
    experiment_plans,
    materialize,
    plot,
    report,
    snapshot,
)
from klip_ppo.cli.eval import eval_command
from klip_ppo.cli.sweep import sweep_command
from klip_ppo.cli.train import train

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="klip — PPO research toolbox (clipping ↔ KL equivalence).",
)
app.command("train", help="Train one Job (config × seed × device).")(train)
app.command("eval", help="Evaluate a saved checkpoint.")(eval_command)
app.command("sweep", help="Run a sweep across many Jobs.")(sweep_command)
app.add_typer(snapshot.app, name="snapshot", help="Freeze / inspect preset snapshots.")
app.add_typer(
    materialize.app,
    name="materialize",
    help="Materialise Python-defined presets to JSON snapshots.",
)
app.add_typer(
    experiment_plans.app,
    name="experiment-plans",
    help="Generate staged benchmark sweep plans.",
)
app.add_typer(plot.app, name="plot", help="Build figures from artifacts.")
app.add_typer(report.app, name="report", help="Build a markdown report.")
app.add_typer(artifacts.app, name="artifacts", help="Browse / gc the artifact tree.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
