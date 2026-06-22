"""
The single PPO training loop.

Variant-agnostic. The only per-variant component handed in is the ``Strategy`` (which
implements the loss). The loop, optimiser, GAE, rollout collection, logging, and
checkpointing are byte-identical across PPO-Clip and the three PPO-KL variants.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from klip_ppo.configs.experiment import ExperimentConfig
from klip_ppo.core.checkpoint import CheckpointManager
from klip_ppo.core.distributions import kl_old_new
from klip_ppo.core.evaluation import EvalStats, evaluate_policy
from klip_ppo.core.losses import explained_variance, partition_indices
from klip_ppo.core.networks import ActorCritic
from klip_ppo.core.ppo.diagnostic_metrics import (
    BETA_QUANTILES,
    BETA_SAMPLE_DIAGNOSTIC_PREFIXES,
)
from klip_ppo.core.ppo.strategies.base import (
    PARTITION_LABEL_IN,
    PARTITION_LABEL_KILL,
    PARTITION_LABEL_PASS,
    PARTITION_LABEL_UNCLASSIFIED,
    EpochAggregate,
    Strategy,
    StrategyOutputs,
    _encode_partition_labels,
)
from klip_ppo.core.rollout import Collector, PPOMinibatch, RolloutBatch
from klip_ppo.core.run_context import RunContext
from klip_ppo.utils.logging import EpochParquetWriter

if TYPE_CHECKING:
    from klip_ppo.utils.logging import Logger


@dataclass(frozen=True)
class RunResult:
    """Summary returned by ``PPOTrainer.run``."""

    run_dir: str
    iterations: int
    env_steps: int
    final_return: float | None
    exit_status: str


@dataclass(frozen=True)
class RolloutTrainStats:
    """Training diagnostics for one rollout update."""

    aggregate: EpochAggregate
    early_stopped: bool
    epoch_records: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    sample_quantiles: dict[str, float | None] = field(default_factory=dict)


def _normalise(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / (t.std() + 1e-8)


class PPOTrainer:
    def __init__(
        self,
        *,
        cfg: ExperimentConfig,
        ctx: RunContext,
        model: ActorCritic,
        collector: Collector,
        strategy: Strategy,
        optim: Optimizer,
        scheduler: LRScheduler | None,
        logger: Logger,
        ckpt: CheckpointManager,
    ) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.model = model
        self.collector = collector
        self.strategy = strategy
        self.optim = optim
        self.scheduler = scheduler
        self.logger = logger
        self.ckpt = ckpt
        self._rng = torch.Generator(device=ctx.device)
        self._rng.manual_seed(ctx.seed)
        self._epoch_writer: EpochParquetWriter | None = None
        if self.cfg.trainer.diagnostic_mode == "full":
            self._epoch_writer = EpochParquetWriter(
                path=ctx.run_dir / "metrics" / "epochs.parquet"
            )

    def run(self) -> RunResult:
        env_steps_per_iter = self.collector.num_envs * self.collector.n_steps
        total_iters = max(1, self.cfg.trainer.total_steps // env_steps_per_iter)
        start_wall = time.monotonic()
        env_step = 0
        last_ep_return: float | None = None
        exit_status = "ok"

        try:
            with tqdm(
                range(1, total_iters + 1),
                desc=self.cfg.name,
                unit="iter",
                disable=not sys.stderr.isatty(),
            ) as progress:
                for it in progress:
                    rollout, ep_stats = self.collector.collect(self.model)
                    self._apply_rollout_advantage_norm(rollout)
                    train_stats = self._train_on_rollout(rollout, iteration=it)
                    if self._epoch_writer is not None and train_stats.epoch_records:
                        self._epoch_writer.write_rows(list(train_stats.epoch_records))
                    prev_env_step = env_step
                    env_step += env_steps_per_iter

                    ev = explained_variance(rollout.values, rollout.returns).item()
                    eval_stats = self._maybe_eval(env_step, prev_env_step, iteration=it)
                    row = self._build_log_row(
                        iteration=it,
                        env_step=env_step,
                        wall_s=time.monotonic() - start_wall,
                        train_stats=train_stats,
                        explained_var=ev,
                        ep_stats=ep_stats,
                        eval_stats=eval_stats,
                    )
                    self.logger.log_iteration(row)
                    last_ep_return = ep_stats.mean_return() or last_ep_return
                    progress.set_postfix(
                        step=env_step,
                        return_mean=last_ep_return,
                        kl=row.get("policy/kl/approx"),
                    )

                    self.ckpt.maybe_save_periodic(
                        every_steps=self.cfg.trainer.checkpoint_every_steps,
                        env_step=env_step,
                        prev_env_step=prev_env_step,
                        iteration=it,
                        model=self.model,
                        optim=self.optim,
                        scheduler=self.scheduler,
                        strategy=self.strategy,
                        collector_state=self.collector.state_dict(),
                    )

                    if self.scheduler is not None:
                        self.scheduler.step()
        except KeyboardInterrupt:
            exit_status = "interrupted"

        if self.cfg.trainer.save_final_checkpoint:
            self.ckpt.save(
                name="final",
                iteration=total_iters,
                env_step=env_step,
                model=self.model,
                optim=self.optim,
                scheduler=self.scheduler,
                strategy=self.strategy,
                collector_state=self.collector.state_dict(),
            )
        self.collector.close()
        if self._epoch_writer is not None:
            self._epoch_writer.close()
        return RunResult(
            run_dir=str(self.ctx.run_dir),
            iterations=total_iters,
            env_steps=env_step,
            final_return=last_ep_return,
            exit_status=exit_status,
        )

    def _apply_rollout_advantage_norm(self, rollout: RolloutBatch) -> None:
        mode = self.cfg.algorithm.advantage_normalization
        if mode == "rollout":
            rollout.advantages = _normalise(rollout.advantages)
        # ``none`` and ``minibatch`` are handled elsewhere; nothing to do here.

    def _maybe_minibatch_normalise(self, mb: PPOMinibatch) -> PPOMinibatch:
        if self.cfg.algorithm.advantage_normalization != "minibatch":
            return mb
        return dataclasses.replace(mb, advantages=_normalise(mb.advantages))

    def _train_on_rollout(
        self, rollout: RolloutBatch, *, iteration: int
    ) -> RolloutTrainStats:
        """
        Run all inner epochs on this rollout.

        Returns the rollout-wide aggregate (mean across every minibatch across every
        epoch). The adaptive-β strategy mutates β in
        ``on_rollout_end(rollout_agg, final_kl=...)``; this happens exactly once per
        rollout update, including when the inner loop breaks early. ``final_kl`` is
        the per-estimator mean KL between rollout policy and post-update policy,
        computed in one no-grad pass over the rollout under the final model state
        (Schulman 2017 §2.3 dual-ascent target).

        Under ``diagnostic_mode = "full"``, also runs one extra no-grad pass over
        the rollout at the end of each inner epoch to compute partition labels
        under a common boundary model state, and emits per-epoch records with
        partition occupancy, epoch-to-epoch migration rate, and per-minibatch
        gradient-norm statistics (both actor-only ``policy_grad_norm`` and global
        ``global_grad_norm``). Records are returned on ``RolloutTrainStats``;
        writing them out is the caller's responsibility.
        """
        rollout_agg = EpochAggregate()
        early_stopped = False
        full_diag = self.cfg.trainer.diagnostic_mode == "full"
        prev_labels: torch.Tensor | None = None
        epoch_records: list[dict[str, Any]] = []
        sample_diagnostics: dict[str, list[torch.Tensor]] = {}
        try:
            for epoch_idx in range(self.cfg.algorithm.epochs):
                epoch_agg = EpochAggregate()
                epoch_policy_grad_norms: list[float] = []
                epoch_global_grad_norms: list[float] = []
                mb_iter = rollout.iter_minibatches_with_indices(
                    self.cfg.algorithm.minibatch_size, generator=self._rng
                )
                for raw_mb, _mb_idx in mb_iter:
                    mb = self._maybe_minibatch_normalise(raw_mb)
                    out = self.strategy.step(mb, self.model)
                    self.optim.zero_grad(set_to_none=True)
                    out.total_loss.backward()
                    policy_grad_norm = _actor_grad_norm(self.model)
                    global_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.algorithm.max_grad_norm
                    )
                    self.optim.step()
                    epoch_agg.update(
                        out,
                        policy_grad_norm=policy_grad_norm,
                        global_grad_norm=float(global_grad_norm),
                    )
                    rollout_agg.update(
                        out,
                        policy_grad_norm=policy_grad_norm,
                        global_grad_norm=float(global_grad_norm),
                    )
                    _collect_sample_diagnostics(sample_diagnostics, out)
                    if full_diag:
                        epoch_policy_grad_norms.append(policy_grad_norm)
                        epoch_global_grad_norms.append(float(global_grad_norm))
                self.strategy.on_epoch_end(epoch_agg)
                if full_diag:
                    boundary_labels = self._compute_boundary_partition_labels(rollout)
                    epoch_records.append(
                        _summarise_epoch(
                            iteration=iteration,
                            epoch=epoch_idx,
                            epoch_labels=boundary_labels,
                            prev_labels=prev_labels,
                            policy_grad_norms=epoch_policy_grad_norms,
                            global_grad_norms=epoch_global_grad_norms,
                            approx_kl_mean=epoch_agg.mean("approx_kl"),
                        )
                    )
                    prev_labels = boundary_labels
                if (
                    self.cfg.algorithm.target_kl_stop is not None
                    and (epoch_agg.mean("approx_kl") or 0.0)
                    > self.cfg.algorithm.target_kl_stop
                ):
                    early_stopped = True
                    break
        finally:
            # Always call once per rollout update, even on early break. The KL
            # used to drive adaptive β is recomputed against the post-update
            # policy over the full rollout (Schulman 2017 §2.3), not the
            # during-training aggregate captured minibatch-by-minibatch.
            final_kl = self._final_policy_kl(rollout)
            self.strategy.on_rollout_end(rollout_agg, final_kl=final_kl)
        return RolloutTrainStats(
            aggregate=rollout_agg,
            early_stopped=early_stopped,
            epoch_records=tuple(epoch_records),
            sample_quantiles=_summarise_sample_quantiles(sample_diagnostics),
        )

    def _final_policy_kl(self, rollout: RolloutBatch) -> dict[str, float]:
        """
        Mean KL between rollout (old) policy and post-update (new) policy.

        One no-grad pass over the full rollout under ``self.model`` in minibatch_size
        chunks. Returns mean values keyed by KL-penalty kind (``"full"``, ``"sample"``,
        ``"k3"``). The adaptive-β controller reads from this dict in ``on_rollout_end``;
        non-adaptive strategies ignore it.

        Recomputing under the final policy (rather than averaging the minibatch-time KL)
        keeps the controller variable consistent with the dual-ascent rule in Schulman
        2017 §2.3.
        """
        n = len(rollout)
        if n == 0:
            return {"full": 0.0, "sample": 0.0, "k3": 0.0}
        minibatch_size = max(1, int(self.cfg.algorithm.minibatch_size))
        device = rollout.obs.device
        full_sum = 0.0
        sample_sum = 0.0
        k3_sum = 0.0
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, n, minibatch_size):
                    end = min(start + minibatch_size, n)
                    idx = torch.arange(start, end, device=device)
                    obs = rollout.obs[idx]
                    actions = rollout.actions[idx]
                    old_logp = rollout.logprobs[idx]
                    old_params = rollout.old_dist_params.index(idx)
                    new_logp, _, _, new_params = self.model.evaluate_actions(
                        obs, actions
                    )
                    log_ratio = new_logp - old_logp
                    ratio = log_ratio.exp()
                    full_kl = kl_old_new(old_params, new_params)
                    sample_kl = old_logp - new_logp
                    k3_kl = (ratio - 1.0) - log_ratio
                    full_sum += float(full_kl.sum())
                    sample_sum += float(sample_kl.sum())
                    k3_sum += float(k3_kl.sum())
        finally:
            self.model.train(was_training)
        return {
            "full": full_sum / n,
            "sample": sample_sum / n,
            "k3": k3_sum / n,
        }

    def _compute_boundary_partition_labels(self, rollout: RolloutBatch) -> torch.Tensor:
        """
        Partition labels for every rollout sample under the current model state.

        One no-grad pass over the full rollout, evaluated at a single (boundary)
        model state. Used to compute the epoch-to-epoch migration rate under
        ``diagnostic_mode = "full"`` as a true boundary diagnostic, rather than
        labelling samples one minibatch at a time while the optimiser is still
        moving the policy mid-epoch.
        """
        n = len(rollout)
        device = rollout.obs.device
        labels = torch.full((n,), -1, dtype=torch.int8, device=device)
        if n == 0:
            return labels
        minibatch_size = max(1, int(self.cfg.algorithm.minibatch_size))
        epsilon = self._partition_epsilon()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, n, minibatch_size):
                    end = min(start + minibatch_size, n)
                    idx = torch.arange(start, end, device=device)
                    obs = rollout.obs[idx]
                    actions = rollout.actions[idx]
                    new_logp, _, _, _ = self.model.evaluate_actions(obs, actions)
                    ratio = (new_logp - rollout.logprobs[idx]).exp()
                    partition = partition_indices(
                        ratio, rollout.advantages[idx], epsilon
                    )
                    labels[idx] = _encode_partition_labels(partition).to(
                        dtype=labels.dtype, device=labels.device
                    )
        finally:
            self.model.train(was_training)
        return labels

    def _partition_epsilon(self) -> float:
        """Clip-band width used for diagnostic partition labelling."""
        algo = self.cfg.algorithm
        for name in ("clip_epsilon", "clip_epsilon_for_diagnostics"):
            value = getattr(algo, name, None)
            if value is not None:
                return float(value)
        return 0.2

    def _maybe_eval(
        self, env_step: int, prev_env_step: int, *, iteration: int
    ) -> EvalStats | None:
        every_steps = self.cfg.trainer.eval_every_steps
        if every_steps is None:
            return None
        if (env_step // every_steps) <= (prev_env_step // every_steps):
            return None
        collector_state = self.collector.state_dict()
        return evaluate_policy(
            self.cfg,
            self.model,
            episodes=self.cfg.trainer.eval_episodes,
            deterministic=self.cfg.trainer.eval_deterministic,
            seed=self.ctx.seed + 10_000 + iteration,
            device=self.ctx.device,
            normalizer_state=collector_state.get("normalizer"),
        )

    def _build_log_row(
        self,
        *,
        iteration: int,
        env_step: int,
        wall_s: float,
        train_stats: RolloutTrainStats,
        explained_var: float,
        ep_stats,
        eval_stats: EvalStats | None = None,
    ) -> dict[str, float | int | None]:
        means = train_stats.aggregate.as_dict()
        policy_grad_norm_var = _variance_from_moments(
            means.get("policy_grad_norm"), means.get("policy_grad_norm_sq")
        )
        global_grad_norm_var = _variance_from_moments(
            means.get("global_grad_norm"), means.get("global_grad_norm_sq")
        )
        migration_summary = _summarise_migrations(train_stats.epoch_records)
        raw_return = ep_stats.mean_raw_return()
        wrapped_return = ep_stats.mean_wrapped_return()
        primary_return = raw_return if raw_return is not None else wrapped_return
        row: dict[str, float | int | None] = {
            "time/iteration": iteration,
            "time/env_step": env_step,
            "time/wall_s": wall_s,
            "train/return/mean": primary_return,
            "train/return/raw_mean": raw_return,
            "train/return/wrapped_mean": wrapped_return,
            "train/return/iqm": ep_stats.iqm_return(),
            "train/episode/len_mean": ep_stats.mean_length(),
            "train/episode/count": ep_stats.n,
            "loss/policy": means.get("policy_loss"),
            "loss/value": means.get("value_loss"),
            "loss/total": means.get("total_loss"),
            "policy/entropy": means.get("entropy"),
            "policy/kl/approx": means.get("approx_kl"),
            "policy/kl/full_mean": means.get("kl_full_mean"),
            "policy/kl/sample_mean": means.get("kl_sample_mean"),
            "policy/clip/fraction": means.get("clip_fraction"),
            "policy/ratio/mean": means.get("ratio_mean"),
            "policy/ratio/min": means.get("ratio_min"),
            "policy/ratio/max": means.get("ratio_max"),
            "policy/ratio/p05": means.get("ratio_p05"),
            "policy/ratio/p95": means.get("ratio_p95"),
            "policy/partition/I_in/fraction": means.get("frac_in_I_in"),
            "policy/partition/I_pass/fraction": means.get("frac_in_I_pass"),
            "policy/partition/I_kill/fraction": means.get("frac_in_I_kill"),
            "policy/partition/I_unclassified/fraction": means.get(
                "frac_in_I_unclassified"
            ),
            "beta/scalar": means.get("beta"),
            "beta/abs_mean/all": means.get("beta_abs_mean_all"),
            "beta/abs_mean/I_kill": means.get("beta_abs_mean_I_kill"),
            "beta/signed_mean/I_kill": means.get("beta_signed_mean_I_kill"),
            "policy/kl/penalty": means.get("kl_penalty"),
            "value/explained_variance": explained_var,
            "optim/policy_grad_norm/mean": means.get("policy_grad_norm"),
            "optim/policy_grad_norm/std": (
                policy_grad_norm_var**0.5 if policy_grad_norm_var is not None else None
            ),
            "optim/policy_grad_norm/var": policy_grad_norm_var,
            "optim/global_grad_norm/mean": means.get("global_grad_norm"),
            "optim/global_grad_norm/std": (
                global_grad_norm_var**0.5 if global_grad_norm_var is not None else None
            ),
            "optim/global_grad_norm/var": global_grad_norm_var,
            "optim/value_grad_norm/mean": None,
            "soft_clip/softness": means.get("soft_clip_softness"),
            "soft_clip/gate/mean/all": means.get("soft_clip_gate_mean"),
            "soft_clip/gate/mean/I_in": means.get("soft_clip_gate_mean_I_in"),
            "soft_clip/gate/mean/I_pass": means.get("soft_clip_gate_mean_I_pass"),
            "soft_clip/gate/mean/I_kill": means.get("soft_clip_gate_mean_I_kill"),
            "soft_clip/gate/mean/I_unclassified": means.get(
                "soft_clip_gate_mean_I_unclassified"
            ),
            "soft_clip/effective_beta/abs_mean/all": means.get(
                "soft_clip_effective_beta_abs_mean_all"
            ),
            "soft_clip/effective_beta/abs_mean/I_kill": means.get(
                "soft_clip_effective_beta_abs_mean_I_kill"
            ),
            "soft_clip/effective_beta/signed_mean/I_kill": means.get(
                "soft_clip_effective_beta_signed_mean_I_kill"
            ),
            "soft_clip/kl_penalty": means.get("soft_clip_kl_penalty"),
            "soft_clip/unclipped_branch_weight/mean/all": means.get(
                "soft_clip_unclipped_branch_weight_mean"
            ),
            "soft_clip/unclipped_branch_weight/mean/I_in": means.get(
                "soft_clip_unclipped_branch_weight_mean_I_in"
            ),
            "soft_clip/unclipped_branch_weight/mean/I_pass": means.get(
                "soft_clip_unclipped_branch_weight_mean_I_pass"
            ),
            "soft_clip/unclipped_branch_weight/mean/I_kill": means.get(
                "soft_clip_unclipped_branch_weight_mean_I_kill"
            ),
            "soft_clip/unclipped_branch_weight/mean/I_unclassified": means.get(
                "soft_clip_unclipped_branch_weight_mean_I_unclassified"
            ),
            "update/steps": train_stats.aggregate.counts,
            "update/early_stopped": float(train_stats.early_stopped),
            "optim/lr": _current_lr(self.optim),
            "diagnostics/migration_rate/mean": migration_summary["migration_rate_mean"],
            "diagnostics/migration_rate/max": migration_summary["migration_rate_max"],
            "diagnostics/policy_grad_norm_var_per_epoch/mean": migration_summary[
                "policy_grad_norm_var_per_epoch_mean"
            ],
            "diagnostics/global_grad_norm_var_per_epoch/mean": migration_summary[
                "global_grad_norm_var_per_epoch_mean"
            ],
        }
        if eval_stats is not None:
            row.update(eval_stats.as_log_row())
        row.update(train_stats.sample_quantiles)
        return row


def _current_lr(optim: Optimizer) -> float:
    for group in optim.param_groups:
        return float(group["lr"])
    return float("nan")


def _variance_from_moments(mean: float | None, mean_sq: float | None) -> float | None:
    if mean is None or mean_sq is None:
        return None
    return max(0.0, mean_sq - mean * mean)


def _collect_sample_diagnostics(
    target: dict[str, list[torch.Tensor]], out: StrategyOutputs
) -> None:
    """Append detached per-sample tensors emitted by a strategy step."""
    for key, values in out.sample_diagnostics.items():
        target.setdefault(key, []).append(values.detach().reshape(-1).cpu())


def _summarise_sample_quantiles(
    samples: dict[str, list[torch.Tensor]],
) -> dict[str, float | None]:
    """Compute fixed quantiles for rollout-level per-sample diagnostics."""
    out: dict[str, float | None] = {}
    levels = np.asarray([level for _, level in BETA_QUANTILES], dtype=np.float64)
    for prefix in BETA_SAMPLE_DIAGNOSTIC_PREFIXES:
        if prefix not in samples:
            continue
        tensors = [values for values in samples[prefix] if values.numel() > 0]
        if not tensors:
            for label, _ in BETA_QUANTILES:
                out[f"{prefix}/{label}"] = None
            continue
        arr = torch.cat(tensors).numpy()
        quantiles = np.quantile(arr, levels)
        for (label, _), value in zip(BETA_QUANTILES, quantiles, strict=True):
            out[f"{prefix}/{label}"] = float(value)
    return out


def _summarise_epoch(
    *,
    iteration: int,
    epoch: int,
    epoch_labels: torch.Tensor,
    prev_labels: torch.Tensor | None,
    policy_grad_norms: list[float],
    global_grad_norms: list[float],
    approx_kl_mean: float | None,
) -> dict[str, Any]:
    """
    One row of per-inner-epoch diagnostics (schema ``EPOCH_PARQUET_SCHEMA``).

    ``epoch_labels`` are partition labels computed once, at the end of the inner epoch,
    against a common post-epoch model state. This makes the epoch-to-epoch migration
    rate a true boundary-to-boundary diagnostic rather than the within-epoch optimizer-
    order artifact produced by labelling minibatch-by-minibatch as gradient steps were
    taken.
    """
    assigned = epoch_labels >= 0
    n_assigned = int(assigned.sum().item())
    denom = n_assigned if n_assigned > 0 else 1
    frac_in = (
        float(((epoch_labels == PARTITION_LABEL_IN) & assigned).sum().item()) / denom
    )
    frac_pass = (
        float(((epoch_labels == PARTITION_LABEL_PASS) & assigned).sum().item()) / denom
    )
    frac_kill = (
        float(((epoch_labels == PARTITION_LABEL_KILL) & assigned).sum().item()) / denom
    )
    frac_unc = (
        float(((epoch_labels == PARTITION_LABEL_UNCLASSIFIED) & assigned).sum().item())
        / denom
    )
    migration_rate: float | None = None
    migration_count: int | None = None
    if prev_labels is not None:
        valid = assigned & (prev_labels >= 0)
        n_valid = int(valid.sum().item())
        if n_valid > 0:
            changed = int(((epoch_labels != prev_labels) & valid).sum().item())
            migration_count = changed
            migration_rate = float(changed) / float(n_valid)
    policy_mean, policy_var = _moments(policy_grad_norms)
    global_mean, global_var = _moments(global_grad_norms)
    return {
        "time/iteration": int(iteration),
        "epoch/index": int(epoch),
        "epoch/samples": int(n_assigned),
        "epoch/partition/I_in/fraction": frac_in,
        "epoch/partition/I_pass/fraction": frac_pass,
        "epoch/partition/I_kill/fraction": frac_kill,
        "epoch/partition/I_unclassified/fraction": frac_unc,
        "epoch/migration/rate": migration_rate,
        "epoch/migration/count": migration_count,
        "epoch/optim/policy_grad_norm/mean": policy_mean,
        "epoch/optim/policy_grad_norm/var": policy_var,
        "epoch/optim/global_grad_norm/mean": global_mean,
        "epoch/optim/global_grad_norm/var": global_var,
        "epoch/policy/kl/approx_mean": approx_kl_mean,
    }


def _moments(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.var())


def _actor_grad_norm(model: ActorCritic) -> float:
    """
    Pre-clip L2 norm of the actor's gradient (policy trunk + head + log_std).

    Computed after ``total_loss.backward()`` but before ``clip_grad_norm_``; because the
    actor and critic have separate trunks in :class:`ActorCritic`, these parameters
    never receive gradient from the value loss, so this norm is a faithful policy-only
    signal rather than the whole-model norm.
    """
    total = 0.0
    for p in model.actor_parameters():
        grad = p.grad
        if grad is None:
            continue
        total += float(grad.detach().pow(2).sum().item())
    return total**0.5


def _summarise_migrations(
    epoch_records: tuple[dict[str, Any], ...],
) -> dict[str, float | None]:
    """Aggregate per-epoch migration / grad-norm-variance stats to one rollout row."""
    rates: list[float] = [
        float(r["epoch/migration/rate"])
        for r in epoch_records
        if r.get("epoch/migration/rate") is not None
    ]
    policy_vars: list[float] = [
        float(r["epoch/optim/policy_grad_norm/var"])
        for r in epoch_records
        if r.get("epoch/optim/policy_grad_norm/var") is not None
    ]
    global_vars: list[float] = [
        float(r["epoch/optim/global_grad_norm/var"])
        for r in epoch_records
        if r.get("epoch/optim/global_grad_norm/var") is not None
    ]
    return {
        "migration_rate_mean": float(np.mean(rates)) if rates else None,
        "migration_rate_max": float(np.max(rates)) if rates else None,
        "policy_grad_norm_var_per_epoch_mean": (
            float(np.mean(policy_vars)) if policy_vars else None
        ),
        "global_grad_norm_var_per_epoch_mean": (
            float(np.mean(global_vars)) if global_vars else None
        ),
    }
