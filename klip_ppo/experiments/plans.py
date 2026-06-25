"""Generate staged sweep plans for the benchmark experiment protocol.

Soft-clipping confirmation note:
    ``p62`` is a **precommitted** confirmation run on the linear-ramp variant at
    softness ``δ = 0.05``. The paper (NeurIPS 2026, §6) describes the linear-ramp
    family as the concrete soft-relaxation instance whose ``δ → 0`` limit
    recovers PPO-Clip, and ``δ = 0.05`` is the precommitted operating point used
    in the screening-to-confirmation handoff. The fixed choice keeps the
    confirmation plan deterministic with respect to the workshop deadline; it is
    not driven by post-hoc screening results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from klip_ppo.configs.sweep import SweepConfig
from klip_ppo.utils.paths import ARTIFACTS_DIR, PROJECT_ROOT, SNAPSHOTS_DIR

DEFAULT_SLOTS: tuple[Mapping[str, Any], ...] = ({"label": "gpu0", "gpu_index": 0},)

DEFAULT_SNAPSHOT_PRESETS_ROOT = SNAPSHOTS_DIR / "presets"
DEFAULT_EXPERIMENT_PLAN_DIR = ARTIFACTS_DIR / "experiment_plans"

SEED0: tuple[int, ...] = (0,)
REST_SEEDS: tuple[int, ...] = (1, 2, 3, 4)
ALL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

SMOKE_OVERRIDES: tuple[str, ...] = (
    "trainer.total_steps=256",
    "rollout.num_envs=2",
    "rollout.n_steps=32",
)

CORE_MUJOCO_ENVS: tuple[str, ...] = (
    "hopper",
    "humanoid",
    "halfcheetah",
    "walker2d",
    "ant",
)
SOFT_CLIP_ENVS: tuple[str, ...] = ("cartpole", "hopper", "halfcheetah")
SOFT_CLIP_MUJOCO_ENVS: tuple[str, ...] = ("hopper", "halfcheetah")
SOFT_CLIP_METHODS: tuple[str, ...] = ("linear_ramp", "sigmoid", "soft_min")
SOFT_CLIP_SOFTNESS_SLUGS: tuple[str, ...] = ("0p01", "0p03", "0p05", "0p1")
SOFT_CLIP_CONFIRM_SOFTNESS_SLUG: str = "0p05"
SOFT_CLIP_BASELINE_ALGOS: tuple[str, ...] = ("ppo_clip", "ppo_kl_per_sample")
DIAGNOSTIC_OVERRIDES: tuple[str, ...] = ("trainer.diagnostic_mode=full",)


@dataclass(frozen=True)
class WrittenExperimentPlan:
    """Summary of one generated experiment sweep plan."""

    path: Path
    name: str
    preset_count: int
    seed_count: int
    job_count: int


def build_default_experiment_plan_payloads(
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_PRESETS_ROOT,
    slots: Sequence[Mapping[str, Any]] = DEFAULT_SLOTS,
    concurrency: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """
    Build the default staged benchmark sweep-plan payloads.

    ``slots`` is a list of slot dicts (``label`` required, ``gpu_index`` optional).
    Multiple slots may share a ``gpu_index`` to over-subscribe one device.
    ``concurrency`` defaults to ``len(slots)``.
    """
    if not snapshot_root.exists():
        raise FileNotFoundError(
            f"snapshot root does not exist: {snapshot_root}; "
            "run `pixi run materialize` first"
        )
    if not slots:
        raise ValueError("slots must contain at least one entry.")
    effective_concurrency = len(slots) if concurrency is None else concurrency

    cc_baselines = _group_paths(snapshot_root, "cc-baselines")
    mujoco_core = _env_paths(snapshot_root, "mujoco-baselines", CORE_MUJOCO_ENVS)
    box2d = _group_paths(snapshot_root, "box2d-baselines")
    cc_sweeps = _group_paths(snapshot_root, "cc-sweeps")
    mujoco_sweeps = _group_paths(snapshot_root, "mujoco-sweeps")
    all_paths = cc_baselines + mujoco_core + box2d + cc_sweeps + mujoco_sweeps

    plan_specs = (
        (
            "p00_launch_smoke_all_seed0_256steps",
            all_paths,
            SEED0,
            SMOKE_OVERRIDES,
        ),
        ("p01_all_presets_seed0_full", all_paths, SEED0, ()),
        ("p10_core_headline_seeds1_4", cc_baselines + mujoco_core, REST_SEEDS, ()),
        ("p20_tuning_sweeps_seeds1_4", cc_sweeps + mujoco_sweeps, REST_SEEDS, ()),
        ("p30_box2d_external_validity_seeds1_4", box2d, REST_SEEDS, ()),
        ("p90_everything_all5_from_scratch", all_paths, ALL_SEEDS, ()),
    )

    return tuple(
        _build_plan_payload(
            name=name,
            paths=paths,
            seeds=seeds,
            overrides=overrides,
            slots=slots,
            concurrency=effective_concurrency,
        )
        for name, paths, seeds, overrides in plan_specs
    )


def build_soft_clipping_experiment_plan_payloads(
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_PRESETS_ROOT,
    slots: Sequence[Mapping[str, Any]] = DEFAULT_SLOTS,
    concurrency: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """
    Build workshop-specific soft-clipping sweep plans.

    The plans are intentionally separate from the default benchmark protocol so
    exploratory soft-clipping runs do not silently expand the core headline suite.
    """
    if not snapshot_root.exists():
        raise FileNotFoundError(
            f"snapshot root does not exist: {snapshot_root}; "
            "run `pixi run materialize` first"
        )
    if not slots:
        raise ValueError("slots must contain at least one entry.")
    effective_concurrency = len(slots) if concurrency is None else concurrency

    smoke_paths = _soft_clip_paths(
        snapshot_root,
        env_slugs=("cartpole", "hopper"),
        methods=SOFT_CLIP_METHODS,
        softness_slugs=SOFT_CLIP_SOFTNESS_SLUGS,
    )
    screening_paths = _soft_clip_paths(
        snapshot_root,
        env_slugs=SOFT_CLIP_ENVS,
        methods=SOFT_CLIP_METHODS,
        softness_slugs=SOFT_CLIP_SOFTNESS_SLUGS,
    ) + _soft_clip_baseline_paths(snapshot_root, SOFT_CLIP_ENVS)
    diagnostic_paths = _soft_clip_paths(
        snapshot_root,
        env_slugs=SOFT_CLIP_MUJOCO_ENVS,
        methods=("linear_ramp",),
        softness_slugs=SOFT_CLIP_SOFTNESS_SLUGS,
    ) + _soft_clip_baseline_paths(snapshot_root, SOFT_CLIP_MUJOCO_ENVS)
    confirmation_paths = _soft_clip_paths(
        snapshot_root,
        env_slugs=SOFT_CLIP_MUJOCO_ENVS,
        methods=("linear_ramp",),
        softness_slugs=(SOFT_CLIP_CONFIRM_SOFTNESS_SLUG,),
    ) + _soft_clip_baseline_paths(snapshot_root, SOFT_CLIP_MUJOCO_ENVS)

    plan_specs = (
        (
            "p50_workshop_soft_clip_smoke_seed0_256steps",
            smoke_paths,
            SEED0,
            SMOKE_OVERRIDES,
        ),
        (
            "p60_workshop_soft_clip_screen_seeds0_2",
            screening_paths,
            (0, 1, 2),
            (),
        ),
        (
            "p61_workshop_soft_clip_diagnostics_linear_ramp_seeds0_2",
            diagnostic_paths,
            (0, 1, 2),
            DIAGNOSTIC_OVERRIDES,
        ),
        (
            "p62_workshop_soft_clip_confirm_precommit_linear_ramp_s0p05_all5",
            confirmation_paths,
            ALL_SEEDS,
            (),
        ),
    )

    return tuple(
        _build_plan_payload(
            name=name,
            paths=paths,
            seeds=seeds,
            overrides=overrides,
            slots=slots,
            concurrency=effective_concurrency,
        )
        for name, paths, seeds, overrides in plan_specs
    )


def write_default_experiment_plans(
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_PRESETS_ROOT,
    output_dir: Path = DEFAULT_EXPERIMENT_PLAN_DIR,
    slots: Sequence[Mapping[str, Any]] = DEFAULT_SLOTS,
    concurrency: int | None = None,
    dry_run: bool = False,
) -> tuple[WrittenExperimentPlan, ...]:
    """Write the default staged benchmark sweep plans as YAML files."""
    payloads = build_default_experiment_plan_payloads(
        snapshot_root=snapshot_root,
        slots=slots,
        concurrency=concurrency,
    )
    return _write_plan_payloads(payloads, output_dir=output_dir, dry_run=dry_run)


def write_soft_clipping_experiment_plans(
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_PRESETS_ROOT,
    output_dir: Path = DEFAULT_EXPERIMENT_PLAN_DIR,
    slots: Sequence[Mapping[str, Any]] = DEFAULT_SLOTS,
    concurrency: int | None = None,
    dry_run: bool = False,
) -> tuple[WrittenExperimentPlan, ...]:
    """Write workshop-specific soft-clipping sweep plans as YAML files."""
    payloads = build_soft_clipping_experiment_plan_payloads(
        snapshot_root=snapshot_root,
        slots=slots,
        concurrency=concurrency,
    )
    return _write_plan_payloads(payloads, output_dir=output_dir, dry_run=dry_run)


def _write_plan_payloads(
    payloads: tuple[dict[str, Any], ...], *, output_dir: Path, dry_run: bool
) -> tuple[WrittenExperimentPlan, ...]:
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    written: list[WrittenExperimentPlan] = []
    for payload in payloads:
        name = str(payload["name"])
        path = output_dir / f"{name}.yaml"
        if not dry_run:
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
        jobs = payload["jobs"]
        seeds = payload["seeds"]
        if not isinstance(jobs, list) or not isinstance(seeds, list):
            raise TypeError("internal error: generated plan has invalid jobs/seeds")
        written.append(
            WrittenExperimentPlan(
                path=path,
                name=name,
                preset_count=len({str(job["config_path"]) for job in jobs}),
                seed_count=len(seeds),
                job_count=len(jobs),
            )
        )
    return tuple(written)


def _group_paths(snapshot_root: Path, group: str) -> tuple[Path, ...]:
    return tuple(sorted((snapshot_root / group).glob("*.json")))


def _env_paths(
    snapshot_root: Path,
    group: str,
    env_slugs: tuple[str, ...],
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in _group_paths(snapshot_root, group)
        if any(path.stem.startswith(f"{slug}__") for slug in env_slugs)
    )


def _job_label(path: Path) -> str:
    return f"{path.parent.name}__{path.stem}"


def _soft_clip_paths(
    snapshot_root: Path,
    *,
    env_slugs: tuple[str, ...],
    methods: tuple[str, ...],
    softness_slugs: tuple[str, ...],
) -> tuple[Path, ...]:
    paths = _group_paths(snapshot_root, "cc-soft-clipping") + _group_paths(
        snapshot_root, "mujoco-soft-clipping"
    )
    selected: list[Path] = []
    for path in paths:
        parsed = _parse_soft_clip_stem(path.stem)
        if parsed is None:
            continue
        env_slug, method, softness_slug = parsed
        if (
            env_slug in env_slugs
            and method in methods
            and softness_slug in softness_slugs
        ):
            selected.append(path)
    return tuple(sorted(selected))


def _parse_soft_clip_stem(stem: str) -> tuple[str, str, str] | None:
    if "__ppo_soft_clip__" not in stem:
        return None
    env_slug, tail = stem.split("__ppo_soft_clip__", maxsplit=1)
    method, sep, softness_slug = tail.rpartition("_s")
    if sep != "_s" or not method or not softness_slug:
        return None
    return env_slug, method, softness_slug


def _soft_clip_baseline_paths(
    snapshot_root: Path, env_slugs: tuple[str, ...]
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for env_slug in env_slugs:
        group = "cc-baselines" if env_slug == "cartpole" else "mujoco-baselines"
        for algo in SOFT_CLIP_BASELINE_ALGOS:
            path = snapshot_root / group / f"{env_slug}__{algo}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"soft-clipping plan requires baseline snapshot: {path}"
                )
            paths.append(path)
    return tuple(paths)


def _build_plan_payload(
    *,
    name: str,
    paths: tuple[Path, ...],
    seeds: tuple[int, ...],
    overrides: tuple[str, ...],
    slots: Sequence[Mapping[str, Any]],
    concurrency: int,
) -> dict[str, Any]:
    if not paths:
        raise ValueError(
            f"plan {name!r} has no matching snapshots; "
            "run `pixi run materialize` first or check --snapshot-root"
        )

    jobs: list[dict[str, Any]] = []
    for config_path in paths:
        for seed in seeds:
            job: dict[str, Any] = {
                "config_path": _portable_config_path(config_path),
                "label": _job_label(config_path),
                "seed": seed,
            }
            if overrides:
                job["overrides"] = list(overrides)
            jobs.append(job)

    payload: dict[str, Any] = {
        "name": name,
        "seeds": list(seeds),
        "concurrency": concurrency,
        "slots": [dict(slot) for slot in slots],
        "jobs": jobs,
    }
    SweepConfig.model_validate(payload)
    return payload


def _portable_config_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
