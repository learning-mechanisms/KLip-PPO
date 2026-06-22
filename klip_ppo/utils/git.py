"""Git introspection used to stamp every run with reproducibility metadata."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from klip_ppo.utils.paths import PROJECT_ROOT

_DIFF_TRUNCATE_BYTES = 64 * 1024


@dataclass(frozen=True)
class GitState:
    commit: str
    short_commit: str
    branch: str | None
    dirty: bool
    diff_truncated: str | None


def _run(cmd: list[str], cwd: Path) -> str:
    return subprocess.run(
        cmd, cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def _try_run(cmd: list[str], cwd: Path) -> str | None:
    try:
        return _run(cmd, cwd)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def read_git_state(cwd: Path = PROJECT_ROOT) -> GitState:
    """
    Return commit, branch, and dirty status of ``cwd``.

    If git is not available or ``cwd`` is not a git repo, returns a sentinel state with
    commit ``"unknown"`` and ``dirty=True``.
    """
    commit = _try_run(["git", "rev-parse", "HEAD"], cwd)
    if commit is None:
        return GitState(
            commit="unknown",
            short_commit="unknown",
            branch=None,
            dirty=True,
            diff_truncated=None,
        )

    branch_raw = _try_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    branch = branch_raw if branch_raw and branch_raw != "HEAD" else None

    status = _try_run(["git", "status", "--porcelain"], cwd) or ""
    dirty = bool(status.strip())

    diff_truncated: str | None = None
    if dirty:
        diff = _try_run(["git", "diff", "--no-color", "HEAD"], cwd) or ""
        if len(diff) > _DIFF_TRUNCATE_BYTES:
            diff_truncated = (
                diff[:_DIFF_TRUNCATE_BYTES]
                + f"\n... <truncated, {len(diff)} bytes total>"
            )
        else:
            diff_truncated = diff or None

    return GitState(
        commit=commit,
        short_commit=commit[:7],
        branch=branch,
        dirty=dirty,
        diff_truncated=diff_truncated,
    )
