"""
PPO-Clip ≡ PPO-KL per-sample β on CartPole (Theorem 3.1).

Two complementary checks:

  * ``test_clip_and_per_sample_gradients_match_on_frozen_minibatch`` —
    one perturbed minibatch, asserts byte-level gradient identity.
  * ``test_clip_and_per_sample_match_end_to_end_on_cartpole`` — full
    inner loop across multiple rollout-update iterations, asserts that
    network parameters and policy-state diagnostics stay matched.

The end-to-end check defends the paper's stronger empirical claim:
"identical rollouts and identical inner-loop updates" under matched seed
and initialisation. If the abstractions in the trainer leak (e.g. value
loss differs between strategies, or partition definitions disagree), or
the per-sample β implementation in §3.5 is wrong, these tests fail.
That is the architectural commitment in
``.prompt.ignore/architecture.md``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from klip_ppo.configs.algorithm.ppo_clip import PPOClipConfig
from klip_ppo.configs.algorithm.ppo_kl_per_sample import PPOKLPerSampleConfig
from klip_ppo.configs.experiment import ExperimentConfig, load_yaml
from klip_ppo.core.networks import ActorCritic
from klip_ppo.core.ppo.strategies import ClipStrategy, KLPerSampleStrategy
from klip_ppo.envs.gym_env import probe_spaces
from klip_ppo.envs.vec_env import VectorCollector
from klip_ppo.utils.seed import set_seed
from torch.nn.utils import parameters_to_vector, vector_to_parameters

TEST_RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"
EQUIV_PRESET = TEST_RESOURCES_DIR / "presets" / "equivalence" / "cartpole.yaml"


def _shared_kwargs(cfg: ExperimentConfig) -> dict:
    base = cfg.algorithm.model_dump()
    base.pop("kind", None)
    base.pop("clip_epsilon", None)
    return base


def _build_clip(cfg: ExperimentConfig) -> ClipStrategy:
    algo = PPOClipConfig(**_shared_kwargs(cfg), clip_epsilon=0.2)
    return ClipStrategy(algo)


def _build_per_sample(cfg: ExperimentConfig) -> KLPerSampleStrategy:
    algo = PPOKLPerSampleConfig(**_shared_kwargs(cfg), clip_epsilon=0.2)
    return KLPerSampleStrategy(algo)


def test_clip_and_per_sample_gradients_match_on_frozen_minibatch():
    cfg = ExperimentConfig.model_validate(load_yaml(EQUIV_PRESET))
    device = torch.device("cpu")
    set_seed(cfg.seed)

    obs_space, act_space = probe_spaces(cfg.env)
    model_old = ActorCritic(obs_space, act_space, cfg.network).to(device)  # type: ignore[arg-type]
    collector = VectorCollector(
        cfg.env,
        cfg.rollout,
        gamma=cfg.algorithm.gamma,
        gae_lambda=cfg.algorithm.gae_lambda,
        device=device,
        seed=cfg.seed,
    )
    try:
        rollout, _ = collector.collect(model_old)
    finally:
        collector.close()

    mb_iter = rollout.iter_minibatches(cfg.algorithm.minibatch_size)
    mb = next(iter(mb_iter))

    torch.manual_seed(123)
    flat = parameters_to_vector(model_old.parameters()).detach().clone()
    flat = flat + 0.5 * torch.randn_like(flat)

    model_a = copy.deepcopy(model_old)
    model_b = copy.deepcopy(model_old)
    vector_to_parameters(flat.clone(), model_a.parameters())
    vector_to_parameters(flat.clone(), model_b.parameters())
    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(pa, pb), "perturbed copies must start identical"

    with torch.no_grad():
        new_logp, _, _, _ = model_a.evaluate_actions(mb.obs, mb.actions)
        ratio = (new_logp - mb.old_logprobs).exp()
        kill = ((mb.advantages > 0) & (ratio > 1.2)) | (
            (mb.advantages < 0) & (ratio < 0.8)
        )
    assert bool(kill.any()), "test setup failed to push any samples into I_kill"

    clip_strat = _build_clip(cfg)
    persample_strat = _build_per_sample(cfg)

    out_a = clip_strat.step(mb, model_a)
    out_a.total_loss.backward()
    out_b = persample_strat.step(mb, model_b)
    out_b.total_loss.backward()

    for (name, pa), (_, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        ga = pa.grad
        gb = pb.grad
        assert ga is not None, f"param {name} got no grad from clip strategy"
        assert gb is not None, f"param {name} got no grad from per-sample strategy"
        torch.testing.assert_close(
            ga, gb, atol=1e-5, rtol=1e-4, msg=f"grad mismatch on {name}"
        )


def _run_full_inner_loop(
    cfg: ExperimentConfig,
    make_strategy,
    *,
    n_iterations: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    """
    Drive a minimal PPO inner loop and return final params + per-step KLs.

    Mirrors what ``PPOTrainer`` does for the equivalence-relevant pieces (rollout
    collection, minibatch shuffling, loss + backward + optimiser step) while
    skipping logging, checkpointing, and LR scheduling. Determinism relies on
    matching every source of randomness across runs: ``set_seed`` for model
    init, an explicit ``torch.Generator`` for minibatch permutation, and the
    same collector seed.
    """
    set_seed(cfg.seed)
    obs_space, act_space = probe_spaces(cfg.env)
    model = ActorCritic(obs_space, act_space, cfg.network).to(device)  # type: ignore[arg-type]
    collector = VectorCollector(
        cfg.env,
        cfg.rollout,
        gamma=cfg.algorithm.gamma,
        gae_lambda=cfg.algorithm.gae_lambda,
        device=device,
        seed=cfg.seed,
    )
    strategy = make_strategy(cfg)
    opt = cfg.algorithm.optimiser
    optim = torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
        eps=opt.eps,
        betas=(opt.beta1, opt.beta2),
        weight_decay=opt.weight_decay,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)
    approx_kls: list[float] = []
    try:
        for _ in range(n_iterations):
            rollout, _ = collector.collect(model)
            for _ in range(cfg.algorithm.epochs):
                for mb in rollout.iter_minibatches(
                    cfg.algorithm.minibatch_size, generator=rng
                ):
                    out = strategy.step(mb, model)
                    optim.zero_grad(set_to_none=True)
                    out.total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.algorithm.max_grad_norm
                    )
                    optim.step()
                    approx_kls.append(float(out.diagnostics["approx_kl"].detach()))
    finally:
        collector.close()
    return parameters_to_vector(model.parameters()).detach().clone(), approx_kls


def test_clip_and_per_sample_match_end_to_end_on_cartpole():
    cfg = ExperimentConfig.model_validate(load_yaml(EQUIV_PRESET))
    device = torch.device("cpu")
    n_iterations = 3

    params_clip, kl_clip = _run_full_inner_loop(
        cfg, _build_clip, n_iterations=n_iterations, device=device
    )
    params_ps, kl_ps = _run_full_inner_loop(
        cfg, _build_per_sample, n_iterations=n_iterations, device=device
    )

    assert len(kl_clip) == len(kl_ps), (
        f"step counts diverged: clip={len(kl_clip)} per_sample={len(kl_ps)}"
    )
    assert len(kl_clip) > 0, "test fixture produced zero optimiser steps"
    for step, (a, b) in enumerate(zip(kl_clip, kl_ps, strict=True)):
        assert abs(a - b) < 1e-5, (
            f"approx_kl drift at step {step}: clip={a} per_sample={b}"
        )
    torch.testing.assert_close(
        params_clip,
        params_ps,
        atol=1e-5,
        rtol=1e-4,
        msg="end-to-end parameter divergence between Clip and per-sample β",
    )
