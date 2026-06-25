"""Build a markdown report from the artifact tree."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from klip_ppo.research.aggregation import load_runs
from klip_ppo.research.tables import final_returns, partition_stats


def build_markdown_report(runs_root: Path, out: Path) -> Path:
    df = load_runs(runs_root)
    lines: list[str] = []
    lines.append("# klip-ppo Empirical Report")
    lines.append("")
    lines.append(f"_Generated {datetime.now(UTC).isoformat()}._")
    lines.append("")
    if df.empty:
        lines.append("No runs found under `" + str(runs_root) + "`.")
        out.write_text("\n".join(lines) + "\n")
        return out

    n_runs = df.groupby(["env", "algo", "seed"]).ngroups
    lines.append(f"Runs aggregated: **{n_runs}**.")
    lines.append("")
    lines.append("## Final returns")
    lines.append("")
    lines.append(final_returns(df).to_markdown(index=False))
    lines.append("")
    lines.append("## Partition occupancy (mean over training)")
    lines.append("")
    pstats = partition_stats(df)
    if not pstats.empty:
        lines.append(pstats.to_markdown(index=False))
    else:
        lines.append("(no partition columns logged)")
    out.write_text("\n".join(lines) + "\n")
    return out
