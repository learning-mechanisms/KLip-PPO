"""In-process (local) execution backend."""

from __future__ import annotations

import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch

from klip_ppo.configs.algorithm.base import PPOAlgoConfigBase
from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.configs.snapshot import ExecutionInfo, GitInfo, SnapshotMetadata
from klip_ppo.core.checkpoint import CheckpointManager
from klip_ppo.core.networks import ActorCritic
from klip_ppo.core.ppo.strategies import build_strategy
from klip_ppo.core.ppo.trainer import PPOTrainer
from klip_ppo.core.run_context import RunContext
from klip_ppo.envs.gym_env import probe_spaces
from klip_ppo.envs.vec_env import VectorCollector
from klip_ppo.runtime.base import RunResult
from klip_ppo.utils import paths
from klip_ppo.utils.git import read_git_state
from klip_ppo.utils.ids import run_dir as build_run_dir_path
from klip_ppo.utils.ids import utc_timestamp
from klip_ppo.utils.lockfile import pixi_lock_sha256
from klip_ppo.utils.log import configure_logging, get_logger, shutdown_logging
from klip_ppo.utils.logging import (
    CompositeLogger,
    Logger,
    ParquetLogger,
    StdLogger,
    WandbLogger,
)
from klip_ppo.utils.seed import set_seed
from klip_ppo.utils.snapshot import (
    build_metadata,
    finalise_metadata,
    write_metadata,
    write_snapshot,
)
from klip_ppo.utils.torch_utils import deterministic_mode, enable_tf32, pick_device
from klip_ppo.utils.wandb_completion import (
    FinishedAtTargetSteps,
    WandbCompletionIndex,
    effective_training_env_steps,
)
from klip_ppo.utils.wandb_identity import wandb_group, wandb_run_name


def worker_main(
    cfg: ExperimentConfig,
    *,
    seed: int,
    input_yaml_path: Path | None = None,
    allow_overwrite: bool = False,
    execution: ExecutionInfo | None = None,
    source_git: GitInfo | None = None,
    source_identity: str | None = None,
    skip_if_complete: bool = False,
) -> RunResult:
    """
    Execute one Job in the current process.

    The CLI ``klip train`` is a thin shim around this. The Sweep runner subprocess
    invokes the same CLI for each Job, which in turn lands here.

    When ``skip_if_complete`` is true, query WandB for a finished run at the job's
    ``(group, seed)`` reaching the trainer's effective final env-step count and short-
    circuit before any expensive setup (env build, model build, run directory creation).
    """
    if skip_if_complete and _wandb_run_already_complete(
        cfg, seed=seed, source_identity=source_identity
    ):
        return RunResult(
            run_dir=Path(""),
            iterations=0,
            env_steps=0,
            final_return=None,
            exit_status="skipped",
        )

    logger: Logger | None = None
    started_at = datetime.now(UTC)
    device = pick_device(cfg.runtime.device)
    deterministic_mode(cfg.runtime.deterministic)
    if cfg.runtime.cudnn_benchmark and not cfg.runtime.deterministic:
        enable_tf32()
    if cfg.runtime.num_threads is not None:
        torch.set_num_threads(int(cfg.runtime.num_threads))

    set_seed(seed)
    local_git = read_git_state()
    git_commit = source_git.commit if source_git is not None else local_git.commit
    git_dirty = source_git.dirty if source_git is not None else local_git.dirty
    git_short = git_commit[:7] if git_commit and git_commit != "unknown" else "unknown"
    run_dir = build_run_dir_path(
        artifacts_root=paths.ARTIFACTS_DIR,
        experiment_name=cfg.name,
        algo_kind=cfg.algorithm.kind,
        env_id=cfg.env.id,
        seed=seed,
        timestamp=utc_timestamp(started_at),
        git_short=git_short,
    )
    _prepare_run_dir(run_dir, allow_overwrite=allow_overwrite)
    configure_logging(
        plain_log_file=run_dir / "logs" / "console.log",
        json_log_file=run_dir / "logs" / "events.jsonl",
    )
    app_log = get_logger(__name__).bind(
        run_dir=str(run_dir),
        experiment=cfg.name,
        env_id=cfg.env.id,
        algorithm=cfg.algorithm.kind,
        seed=seed,
    )
    app_log.info(
        "run_created",
        device=str(device),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    write_snapshot(run_dir / "snapshot.json", cfg)
    if input_yaml_path is not None and input_yaml_path.exists():
        shutil.copy2(input_yaml_path, run_dir / "config.input.yaml")
    metadata = build_metadata(
        seed=seed,
        started_at=started_at,
        execution=execution,
        source_git=source_git or local_git,
        effective_device=str(device),
    )
    write_metadata(run_dir / "metadata.json", metadata)

    ctx = RunContext(
        device=device,
        seed=seed,
        run_dir=run_dir,
        git_commit=git_commit,
        git_dirty=git_dirty,
        pixi_lock_sha=pixi_lock_sha256(),
        started_at=started_at,
    )

    obs_space, act_space = probe_spaces(cfg.env)
    app_log.info(
        "environment_ready",
        observation_space=str(obs_space),
        action_space=str(act_space),
    )
    model = ActorCritic(obs_space, act_space, cfg.network).to(device)  # type: ignore[arg-type]
    app_log.info("model_ready")

    collector = VectorCollector(
        cfg.env,
        cfg.rollout,
        gamma=cfg.algorithm.gamma,
        gae_lambda=cfg.algorithm.gae_lambda,
        device=device,
        seed=seed,
    )

    strategy = build_strategy(cfg.algorithm)
    optim = _build_optimiser(model, cfg.algorithm)
    scheduler = _build_scheduler(optim, cfg)
    logger = _build_logger(
        cfg,
        run_dir,
        seed=seed,
        metadata=metadata,
        source_identity=source_identity,
    )
    ckpt = CheckpointManager(run_dir)

    exit_status = "ok"
    error_message: str | None = None
    last_iteration: int | None = None
    try:
        trainer = PPOTrainer(
            cfg=cfg,
            ctx=ctx,
            model=model,
            collector=collector,
            strategy=strategy,
            optim=optim,
            scheduler=scheduler,
            logger=logger,
            ckpt=ckpt,
        )
        app_log.info(
            "trainer_start",
            total_steps=cfg.trainer.total_steps,
            rollout_num_envs=cfg.rollout.num_envs,
            rollout_n_steps=cfg.rollout.n_steps,
        )
        result = trainer.run()
        last_iteration = result.iterations
        exit_status = result.exit_status
    except Exception as exc:
        exit_status = "error"
        error_message = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc()
        app_log.exception("run_failed", error=str(exc))
        collector.close()
        raise
    finally:
        ended_at = datetime.now(UTC)
        final_meta = finalise_metadata(
            metadata,
            exit_status=exit_status,
            error_message=error_message,
            last_completed_iteration=last_iteration,
            ended_at=ended_at,
        )
        write_metadata(run_dir / "metadata.json", final_meta)
        if exit_status != "error":
            app_log.info(
                "run_finished",
                exit_status=exit_status,
                last_completed_iteration=last_iteration,
            )
        if logger is not None:
            logger.close()
        shutdown_logging()

    return RunResult(
        run_dir=run_dir,
        iterations=result.iterations,
        env_steps=result.env_steps,
        final_return=result.final_return,
        exit_status=result.exit_status,
    )


class LocalRuntime:
    """Adapter so the CLI can dispatch on a Runtime instance."""

    def run_training(
        self,
        cfg: ExperimentConfig,
        *,
        seed: int,
        input_yaml_path: Path | None = None,
        allow_overwrite: bool = False,
        execution: ExecutionInfo | None = None,
        source_git: GitInfo | None = None,
        source_identity: str | None = None,
        skip_if_complete: bool = False,
    ) -> RunResult:
        return worker_main(
            cfg,
            seed=seed,
            input_yaml_path=input_yaml_path,
            allow_overwrite=allow_overwrite,
            execution=execution,
            source_git=source_git,
            source_identity=source_identity,
            skip_if_complete=skip_if_complete,
        )


def _wandb_run_already_complete(
    cfg: ExperimentConfig,
    *,
    seed: int,
    source_identity: str | None,
) -> bool:
    """
    Return True iff WandB already has a finished run for this ``(group, seed)``.

    Requires WandB to be configured on the cfg; raises otherwise so the caller cannot
    accidentally skip the preflight by forgetting to enable WandB.
    """
    wandb_cfg = cfg.logging.wandb
    if wandb_cfg is None:
        raise RuntimeError(
            "skip_if_complete requires logging.wandb to be configured "
            "(set WANDB_PROJECT or pass --wandb-project)."
        )
    group = wandb_group(cfg, source_identity=source_identity)
    index = WandbCompletionIndex(entity=wandb_cfg.entity, project=wandb_cfg.project)
    predicate = FinishedAtTargetSteps(
        target_steps=effective_training_env_steps(
            total_steps=cfg.trainer.total_steps,
            num_envs=cfg.rollout.num_envs,
            n_steps=cfg.rollout.n_steps,
        )
    )
    return index.is_complete(group=group, seed=seed, predicate=predicate)


def _prepare_run_dir(run_dir: Path, *, allow_overwrite: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        if not allow_overwrite:
            raise FileExistsError(
                f"run directory already exists and is non-empty: {run_dir}\n"
                "pass --allow-overwrite to force."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _build_optimiser(
    model: ActorCritic, algo: PPOAlgoConfigBase
) -> torch.optim.Optimizer:
    opt = algo.optimiser
    return torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
        eps=opt.eps,
        betas=(opt.beta1, opt.beta2),
        weight_decay=opt.weight_decay,
    )


def _build_scheduler(
    optim: torch.optim.Optimizer, cfg: ExperimentConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Linear LR anneal from ``lr`` -> 0 across the run (CleanRL convention).

    ``scheduler.step()`` is called once per training iteration by ``PPOTrainer``;
    the factor at iteration ``k`` (0-indexed, before any step()) is ``1 - k/N``,
    so iteration 1 trains at full ``lr`` and iteration ``N`` trains at ``lr/N``.
    """
    if not cfg.algorithm.optimiser.anneal_lr:
        return None
    env_steps_per_iter = cfg.rollout.num_envs * cfg.rollout.n_steps
    total_iters = max(1, cfg.trainer.total_steps // env_steps_per_iter)

    def _factor(step: int) -> float:
        return max(0.0, 1.0 - step / total_iters)

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_factor)


def _build_logger(
    cfg: ExperimentConfig,
    run_dir: Path,
    *,
    seed: int,
    metadata: SnapshotMetadata,
    source_identity: str | None = None,
) -> Logger:
    sinks: list[Logger] = []
    if cfg.logging.stdout:
        sinks.append(StdLogger(log_file=run_dir / "stdout.log"))
    if cfg.logging.parquet:
        sinks.append(
            ParquetLogger(
                path=run_dir / "metrics" / "train.parquet",
                flush_every=int(cfg.logging.parquet_flush_every_iters),
            )
        )
    if cfg.logging.wandb is not None:
        tags = tuple(dict.fromkeys((*cfg.tags, *cfg.logging.wandb.tags)))
        wb_group = wandb_group(cfg, source_identity=source_identity)
        wb_run_name = wandb_run_name(
            cfg,
            seed=seed,
            run_dir=run_dir,
            source_identity=source_identity,
        )
        sinks.append(
            WandbLogger(
                project=cfg.logging.wandb.project,
                run_name=wb_run_name,
                entity=cfg.logging.wandb.entity,
                group=wb_group,
                tags=tags,
                mode=cfg.logging.wandb.mode,
                config=_wandb_config_payload(
                    cfg,
                    run_dir,
                    metadata,
                    wandb_group=wb_group,
                    wandb_run_name=wb_run_name,
                    source_identity=source_identity,
                ),
                notes=cfg.notes or None,
                job_type=cfg.logging.wandb.job_type,
                resume=cfg.logging.wandb.resume,
                run_dir=run_dir,
                upload_artifacts=cfg.logging.wandb.upload_artifacts,
                artifact_aliases=cfg.logging.wandb.artifact_aliases,
            )
        )
    return CompositeLogger(sinks=sinks)


def _wandb_config_payload(
    cfg: ExperimentConfig,
    run_dir: Path,
    metadata: SnapshotMetadata,
    *,
    wandb_group: str,
    wandb_run_name: str,
    source_identity: str | None,
) -> dict[str, object]:
    payload = cfg.model_dump(mode="json")
    run_payload: dict[str, object] = {
        "run_dir": str(run_dir),
        "device": metadata.host.effective_device,
        "git": metadata.git.model_dump(mode="json"),
        "host": metadata.host.model_dump(mode="json"),
        "execution": metadata.execution.model_dump(mode="json"),
        "wandb": {
            "group": wandb_group,
            "run_name": wandb_run_name,
        },
    }
    if source_identity is not None:
        run_payload["source_identity"] = source_identity
    payload["run"] = run_payload
    return payload
