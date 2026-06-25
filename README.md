# KLip-PPO: A per-sample KL perspective on PPO-Clip

[![CI](https://github.com/learning-mechanisms/KLip-PPO/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/learning-mechanisms/KLip-PPO/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/learning-mechanisms/KLip-PPO/ci.yml?branch=main&label=tests)](https://github.com/learning-mechanisms/KLip-PPO/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/learning-mechanisms/KLip-PPO/branch/main/graph/badge.svg)](https://codecov.io/gh/learning-mechanisms/KLip-PPO)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%2B%20CC%20BY%204.0-blue)](#license)
[![W&B Artifacts](https://img.shields.io/badge/W%26B-public%20artifacts-ffbe00)](https://wandb.ai/KLip-PPO/KLip-PPO)

> The gradient of `PPO`'s clipped surrogate is reproduced _exactly_ by a
> `KL`-penalty surrogate whose coefficient varies per sample. So `PPO-Clip` is best
> understood as an adaptive `KL` penalty, not just a ratio-clipping heuristic.

### University of California, Berkeley

- `Riccardo Colletti` 🎓 (\*) riccardo.colletti [at] berkeley.edu
- `Robin Holzinger` 🎓 (\*) robin.holzinger [at] berkeley.edu

(\*) Equal contribution

---

## Contents

1. [Project Summary](#project-summary)
2. [Core Idea](#core-idea)
3. [Core Theory](#core-theory)
4. [Core Objectives](#core-objectives)
5. [Installation](#installation)
6. [Quick start](#quick-start)
7. [Repository layout](#repository-layout)
8. [Running experiments](#running-experiments)
9. [Reproducing the paper](#reproducing-the-paper)
10. [Experiment tracking (Weights & Biases)](#experiment-tracking-weights--biases)
11. [Running on Modal](#running-on-modal)
12. [Algorithmic conventions](#algorithmic-conventions)
13. [Citation](#citation)
14. [License](#license)
15. [Reproducibility checklist](#reproducibility-checklist)

---

## Project Summary

Proximal Policy Optimization (PPO) is the default policy-gradient algorithm for
on-policy reinforcement learning. The literature presents it in two forms that
are usually treated as separate algorithms:

- **PPO-Clip** — clips the importance ratio between the new and old policies inside
  a fixed band.
- **PPO-KL** — adds a Kullback–Leibler penalty between the two policies.

This project shows that the two are the same thing at the gradient level: PPO-Clip's
update is exactly a KL-penalty update whose penalty coefficient is chosen
**per sample**, with a closed form in the importance ratio and the advantage. The
identity holds at every minibatch step and across the whole inner loop, and on five
MuJoCo control benchmarks the two losses produce indistinguishable training curves.

The repository turns that statement into a reproducible research stack: matching
objective kernels, gradient-equivalence tests, instrumented training, a benchmark
suite, and a one-command path from frozen data to the paper figures.

Alongside the code you'll find the paper at
[`paper/KLip-PPO.pdf`](paper/KLip-PPO.pdf) and a companion website in
[`website/`](website/) / [klip-ppo.org](https://klip-ppo.org/). The public
[Weights & Biases project](https://wandb.ai/KLip-PPO/KLip-PPO) hosts the run
histories, configs, logs, metrics, and checkpoints used as reproducibility
artifacts.

## Core Idea

Both PPO-Clip and PPO-KL start from the same importance-weighted surrogate:

```text
w_t = pi_new(a_t | s_t) / pi_old(a_t | s_t)      # per-sample importance ratio
L_surrogate = E[ w_t * A_t ]                      # A_t = advantage estimate
```

PPO-Clip clips `w_t`; PPO-KL adds a KL term. For each sampled transition,
PPO-Clip falls into one of three regions:

| Region   | Condition                                      | Effect on the gradient                                        |
| -------- | ---------------------------------------------- | ------------------------------------------------------------- |
| `I_in`   | ratio inside `[1 − ε, 1 + ε]`                  | gradient flows normally                                       |
| `I_kill` | ratio outside the band, update already helpful | gradient is **suppressed** (clipping has done its job)        |
| `I_pass` | ratio outside the band, but update is harmful  | unclipped term stays **active** (still correcting a bad move) |

At the gradient level, that switching behaviour is matched by a per-sample KL
coefficient:

```text
beta_t = 0           if sample t is in I_in or I_pass
beta_t = -w_t * A_t  if sample t is in I_kill
```

This is **not** the same as standard PPO-KL with one global scalar `beta`. The
implicit per-sample penalty is a step function at the boundary of the trust region —
and the _shape_ of that coefficient is the natural axis for generalising the
algorithm (which is what the soft-clip variant explores).

> **Central takeaway:** treat PPO-Clip as an adaptive penalty mechanism. This repo
> is built to test that claim both numerically (exact gradient equivalence) and
> empirically (matched training curves).

## Core Theory

A few questions drive the experiments:

- Do autograd gradients for PPO-Clip exactly match the adaptive-`beta` PPO-KL
  construction on frozen minibatches? (Yes — guarded by a test, see below.)
- How often do samples enter `I_in`, `I_kill`, and `I_pass` during training?
- How close is the best _scalar_ `beta` to the implied _per-sample_ `beta_t`
  distribution?
- Do killed gradients correlate with lower policy KL, improved stability, or better
  returns?
- Which differences survive between PPO-Clip and PPO-KL once objective gradients,
  advantage normalization, entropy bonuses, and optimizer details are all controlled?

The exact gradient equivalence is enforced by
[`tests/integration/test_equivalence.py`](tests/integration/test_equivalence.py),
which fails if PPO-Clip and adaptive-`beta` PPO-KL gradients diverge beyond
numerical tolerance.

## Core Objectives

All five variants are implemented as side-by-side objective kernels and selected by
a config discriminator. On any given environment they share _exactly_ the same env,
network, rollout, and trainer settings, so differences come only from the objective.

| Config key          | Variant           | What it does                                                                  |
| ------------------- | ----------------- | ----------------------------------------------------------------------------- |
| `ppo_clip`          | PPO-Clip          | The standard clipped surrogate `min(w·A, clip(w)·A)`.                         |
| `ppo_kl_fixed`      | fixed-β PPO-KL    | KL penalty with a constant scalar `beta`.                                     |
| `ppo_kl_adaptive`   | adaptive-β PPO-KL | `beta` adjusted once per rollout to track a KL target.                        |
| `ppo_kl_per_sample` | per-sample PPO-KL | The construction that reproduces PPO-Clip's gradient exactly.                 |
| `ppo_soft_clip`     | soft-clip PPO     | Replaces the hard kill boundary with a smooth gate (the generalisation axis). |

## Installation

The project uses [pixi](https://pixi.dev) for a fully pinned, reproducible
environment (Python 3.12, PyTorch, Gymnasium, MuJoCo, Stable-Baselines3, plotting,
notebooks, tests, and the lint stack).

```bash
# 1. Install pixi if you don't have it.
brew install pixi            # macOS; see pixi.dev for other platforms

# 2. Install the environment and the editable package.
pixi install
pixi run postinstall

# 3. (Optional) Register the git pre-commit hooks.
pixi run pre-commit-install
pixi run pre-commit-run       # run them once over all files

# 4. Confirm the install.
pixi run test                 # full suite
pixi run test-fast            # unit tests only (quick)
```

**Notebooks:** select the kernel at `.pixi/envs/default/bin/python3`.

**Auto-activation (optional):** `brew install direnv && direnv allow` activates the
pixi environment whenever you `cd` into the repo.

## Quick start

Train PPO-Clip on CartPole from a checked-in snapshot — CPU only, under two minutes:

```bash
# 1. Materialise every preset to a frozen JSON snapshot.
pixi run materialize

# 2. Train PPO-Clip on CartPole-v1, seed 0.
pixi run train --from-snapshot \
    configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json --seed 0

# 3. Replay another seed declared on the snapshot.
pixi run train --from-snapshot \
    configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json --seed 1

# 4. Plot learning curves once a few runs exist.
pixi run plot curves --algos ppo_clip --env CartPole-v1
```

Swap the snapshot path to train any other variant or environment, e.g.
`configs/snapshots/presets/mujoco-baselines/hopper__ppo_kl_adaptive.json`.

## Repository layout

```text
klip_ppo/            Python package (the implementation)
├── configs/         Pydantic v2 config models + the algorithm discriminator
├── core/            networks, GAE, rollout buffer, losses, PPO trainer + per-variant strategies
├── envs/            Gymnasium env factory, obs/reward normalisation, vectorised collector
├── runtime/         local + Modal backends and the sweep runner
├── cli/             Typer CLI (train, eval, sweep, snapshot, materialize, plot, report, ...)
├── experiments/     Python source-of-truth for all benchmark and sweep presets
├── research/        artifact aggregation, plotting, table builders, markdown report
└── utils/           paths, seeding, git, snapshots, logging, torch helpers, wandb identity

configs/snapshots/   Frozen JSON snapshots of every preset (checked in, fingerprint-free)
tests/               unit + integration tests (incl. the gradient-equivalence check)
analysis/            reproducible paper figures from frozen datasets
submission/          self-contained artifact bundle that rebuilds the figures offline
paper/               the LaTeX writeup and generated figures
website/             companion website
artifacts/           runtime outputs (gitignored)
```

The `configs/algorithm/` discriminator selects between `ppo_clip`, `ppo_kl_fixed`,
`ppo_kl_adaptive`, `ppo_kl_per_sample`, and `ppo_soft_clip`. Snapshot groups are
`cc-baselines`, `cc-sweeps`, `cc-soft-clipping`, `mujoco-baselines`,
`mujoco-sweeps`, `mujoco-soft-clipping`, and `box2d-baselines`.

## Running experiments

### Presets and snapshots

Every benchmark and sweep is defined once in Python under
[`klip_ppo/experiments/`](klip_ppo/experiments/) (one module per group) and
**materialised** into frozen JSON snapshots under `configs/snapshots/presets/`. The
snapshots are the reproducible source of truth — each envelope declares the seed set
the preset is meant to run over (`seeds: [0, 1, 2, 3, 4]` by default).

```bash
pixi run materialize                       # regenerate every snapshot
klip materialize all --group mujoco-baselines   # or just one group
klip materialize list                      # list the groups
```

Parity across the five variants on a given env (same env / network / rollout /
trainer settings and shared algorithm knobs) is guaranteed by
`klip_ppo/experiments/common.py`, not by hand-copying YAML.

### A single training run

```bash
pixi run train --from-snapshot \
    configs/snapshots/presets/mujoco-baselines/hopper__ppo_kl_adaptive.json --seed 0
```

Each run directory captures everything needed to reproduce it: the input config,
`snapshot.json`, `metadata.json` (incl. git commit), console + structured logs,
per-iteration metrics (JSONL and parquet), and the final checkpoint. Interactive
terminals show progress bars.

### Sweeps and staged experiment plans

Generate validated, staged sweep plans from the snapshots:

```bash
pixi run experiment-plans        # baselines + tuning sweeps
pixi run soft-clipping-plans     # the soft-clip workshop grid
```

These write YAML plans under `artifacts/experiment_plans/`:

| Plan                                   | Scope                                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| `p00_launch_smoke_all_seed0_256steps`  | tiny smoke launch for every preset                                                    |
| `p01_all_presets_seed0_full`           | one full seed for every preset                                                        |
| `p10_core_headline_seeds1_4`           | remaining seeds for CartPole + core MuJoCo (Hopper/Humanoid/HalfCheetah/Walker2d/Ant) |
| `p20_tuning_sweeps_seeds1_4`           | remaining seeds for the tuning sweeps                                                 |
| `p30_box2d_external_validity_seeds1_4` | remaining LunarLander seeds                                                           |
| `p90_everything_all5_from_scratch`     | full fallback plan                                                                    |

`pixi run experiment-plans` is a thin alias for:

```bash
klip experiment-plans generate \
    --snapshot-root configs/snapshots/presets \
    --output-dir artifacts/experiment_plans
```

Useful flags: `--device cpu|mps`, `--gpu-indices 0,1,2,3`, `--jobs-per-device`
(concurrent jobs per device), `--concurrency` (cap on in-flight jobs), and
`--dry-run` (validate and print the summary without writing YAML).

Sweep jobs that omit `seed` expand to the default five seeds `0, 1, 2, 3, 4`; set
`seeds: [...]` on the sweep or `seed` on an individual job to change that. Sweep
`config_path` entries can point at YAML configs or at checked-in snapshot JSON
(JSON jobs replay through the same frozen-snapshot path as `train --from-snapshot`).

### The benchmark suite

The full suite is the CartPole baselines, the MuJoCo locomotion set (Hopper,
Humanoid, HalfCheetah, Walker2d, Ant), LunarLander, the `β` / `kl_target` /
clip-`ε` sweeps on CartPole / Hopper / HalfCheetah, and the soft-clip workshop grid
(`linear_ramp` / `sigmoid` / `soft_min` crossed with four `softness` values).

The protocol uses CleanRL-style MuJoCo defaults for the main locomotion cells
(`num_envs=1`, `n_steps=2048`, `epochs=10`, `minibatch_size=64`, `lr=3e-4`, linear
LR annealing, observation + reward normalisation). Humanoid runs longer (`10M`
steps) so its curves compare with long-horizon baselines. LunarLander uses
`LunarLander-v3` with the RL Zoo recipe (`epochs=10` kept for parity).

## Reproducing the paper

The paper figures rebuild deterministically from **frozen datasets** — no training
or network access required.

```bash
pixi run figures        # analysis/datasets/ -> paper/figures, verified against figures.sha256
scripts/build_paper.sh  # builds the paper PDF and syncs the website copy
pixi run export-data    # (online) refresh the frozen datasets from the pinned runs
```

`analysis/figures.sha256` pins the expected byte-for-byte output, and the build is
checked against it in the tests. See [`analysis/README.md`](analysis/README.md) for
the two-stage (online export / offline render) design.

For a fully self-contained, offline-rebuildable bundle, use the artifact tooling in
[`submission/`](submission/):

```bash
pixi run artifact-validate   # confirm the offline build matches the checksums
pixi run artifact            # assemble dist/artifact
pixi run artifact-zip        # byte-stable dist/artifact.zip
pixi run artifact-unpack     # extract clean and rebuild from scratch
```

## Experiment tracking (Weights & Biases)

Training can mirror the local artifact stream to W&B. Configure it via
`WANDB_PROJECT` / `WANDB_ENTITY`, a `logging.wandb` config block, or CLI overrides:

```bash
pixi run train --from-snapshot \
    configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json \
    --set logging.wandb.project=klip-ppo \
    --set logging.wandb.mode=online
```

The `modal-train` / `modal-sweep` tasks source `.env` first, so W&B credentials can
stay out of the checked-in presets:

```bash
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=klip-ppo
WANDB_MODE=online
```

Each W&B run logs the same per-iteration metrics as `metrics/train.parquet`, using
identical slash-prefixed names in both places — e.g. `train/return/mean`,
`policy/kl/approx`, `policy/clip/fraction`, `policy/partition/I_kill/fraction`,
`beta/per_sample/all/p50`, `optim/policy_grad_norm/mean`, `optim/lr`. On close it
uploads the reproducibility files from the run directory (snapshot, metadata, input
config, logs, metrics, and the final checkpoint when present).

The public artifact table is available at
[wandb.ai/KLip-PPO/KLip-PPO](https://wandb.ai/KLip-PPO/KLip-PPO). It mirrors the
checked-in snapshot identities and seed replicas, so readers can inspect individual
learning curves and download the artifacts behind the frozen datasets.

By default, runs are grouped to match the checked-in snapshot file. Replaying
`cc-baselines/cartpole__ppo_clip.json` uses group `cc-baselines__cartpole__ppo_clip`
with run names `...__seed=<n>__<run-leaf>`. Explicit `logging.wandb.group` /
`logging.wandb.run_name` override the base names, but run names keep the `seed=<n>`
suffix so multi-seed sweeps stay comparable. Reports and plots can also be
published:

```bash
pixi run report build --wandb-project klip-ppo
pixi run plot curves --algos ppo_clip --env CartPole-v1 --wandb-project klip-ppo
```

Run lifecycle logs land in `logs/console.log` and `logs/events.jsonl` per run; local
sweeps additionally write parent lifecycle logs under `artifacts/sweeps/<...>/logs/`
and per-child output in `logs/<label>__seed<n>.log`.

## Running on Modal

Modal uses the same config and artifact layout as local runs — each remote job runs
the single-device trainer in one container and writes to the `klip-ppo-artifacts`
Modal Volume.

```bash
pixi run modal-deploy            # (optional) deploy/update the Modal app
pixi run modal-wandb-secret      # (optional) publish W&B creds from .env as a Modal secret

# CPU smoke job.
pixi run modal-train --from-snapshot \
    configs/snapshots/presets/cc-baselines/cartpole__ppo_clip.json \
    --seed 0 --modal-gpu cpu

# GPU job.
pixi run modal-train --from-snapshot \
    configs/snapshots/presets/mujoco-baselines/hopper__ppo_clip.json \
    --seed 0 --modal-gpu L4

# Pull remote artifacts back into ./artifacts (same runs/sweeps layout).
pixi run modal-pull --all
```

Modal launches refuse dirty git trees by default, so remote `metadata.json` carries
an unambiguous commit hash; pass `--allow-dirty-modal` for exploratory runs (the
truncated diff is recorded in metadata). Functions default to a 24h timeout cap.
When `WANDB_PROJECT` or `KLIP_MODAL_WANDB_SECRET` is set on the submit side, workers
mount the W&B secret named by `KLIP_MODAL_WANDB_SECRET` (default `wandb`).

## Algorithmic conventions

A few choices map to specific equations rather than to convenient code, and matter
for keeping the five variants comparable:

- **`policy/kl/approx`** (logged every iteration) is Schulman's low-variance k3
  estimator `E[(w − 1) − log w]`. It is a _diagnostic_ and the gate for the optional
  `target_kl_stop` early-exit — it is **not** what PPO-KL optimises.
- **`KLFixedStrategy` / `KLAdaptiveStrategy`** optimise `−E[w A] + β · E[KL_penalty]`,
  where `KL_penalty` defaults to the closed-form `KL(π_old || π_new)` (TRPO
  trust-region form). The `kl_penalty` field switches between `"full"`, `"sample"`
  (log-ratio surrogate), and `"k3"` for ablations.
- **`KLPerSampleStrategy`** intentionally uses the sampled log-probability penalty
  `kl_t = log π_old(a_t|s_t) − log π_new(a_t|s_t)`, not the full KL. The equivalence
  proof cancels the _sampled_ surrogate gradient with the _sampled_ penalty term;
  substituting the analytic KL would break that cancellation. Guarded by
  `tests/integration/test_equivalence.py`.
- **`SoftClipStrategy`** replaces the hard `min(w·A, clip(w)·A)` switch with a smooth
  gate at the kill boundary. `method` selects `linear_ramp`, `sigmoid`, or
  `soft_min`; `softness` controls how gradual the boundary is (smaller → closer to
  hard PPO-Clip). Presets sweep `softness ∈ {0.01, 0.03, 0.05, 0.1}` across all three
  methods. Note `sigmoid` has a nonzero inside-band gate tail at a shared `softness`
  while the others don't, so the three methods are not strictly comparable until you
  equalise the inside-band gate mean (`soft_clip/gate/mean/I_in`).
- **Adaptive β** is updated once per rollout update (after all inner epochs), with
  literature defaults `kl_high_ratio=1.5`, `kl_low_ratio=1/1.5`, `beta_inc_factor=2`.
- **`kl_target=0.02`** (not the paper's common `0.01`) is the default for adaptive-KL
  presets: `0.02 ≈ 0.2² / 2`, the second-order KL scale for the default clip radius.
  Sweeps still include `0.003`, `0.01`, `0.03`, the local default `0.02`, and a loose
  `0.1`.
- **`train/return/mean`** reports raw env returns from a `RecordEpisodeStatistics`
  wrapper inserted _before_ reward normalisation, so curves stay comparable to
  published PPO benchmarks even when `normalize_reward=true`. The post-normalisation
  stream is logged as `train/return/wrapped_mean`.
- **`advantage_normalization`** is an explicit `none / rollout / minibatch` knob on
  every algorithm config. Rollout-scope (default) matches the CleanRL/IsaacGym
  convention; minibatch-scope matches SB3.

## Citation

If you use the paper, code, or frozen artifacts, please cite the paper. The
repository also includes [`CITATION.cff`](CITATION.cff), so GitHub's
**Cite this repository** button and citation managers can pick up the same
metadata.

```bibtex
@misc{colletti2026klipppo,
  title = {{KLip-PPO}: A per-sample {KL} perspective on {PPO-Clip}},
  author = {Colletti, Riccardo and Holzinger, Robin},
  year = {2026},
  url = {https://klip-ppo.org/KLip-PPO.pdf},
  note = {Code: \url{https://github.com/learning-mechanisms/KLip-PPO}},
}
```

If you want to cite the repository separately, use:

```bibtex
@misc{colletti2026klipppo_software,
  title = {{KLip-PPO}},
  author = {Colletti, Riccardo and Holzinger, Robin},
  year = {2026},
  url = {https://github.com/learning-mechanisms/KLip-PPO},
}
```

## License

Code is licensed under the [Apache License 2.0](LICENSE). The paper, figures,
website, and documentation are licensed under
[Creative Commons Attribution 4.0 International](LICENSE-paper) where we own the
rights. Third-party references and dependencies remain under their respective
licenses.

## Reproducibility checklist

- [x] fixed random seeds for environment, NumPy, PyTorch, and action sampling
- [x] saved configs for every run
- [x] logged objective components before and after each policy update
- [x] stored frozen minibatches for gradient debugging
- [x] explicit control of advantage normalization, entropy bonus, value-loss weight, optimizer, learning rate, and minibatch order
- [x] tests that fail when PPO-Clip and adaptive-β PPO-KL gradients diverge beyond
      numerical tolerance.
