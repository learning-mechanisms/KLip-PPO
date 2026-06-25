# artifacts/

This directory holds **runtime outputs** of training jobs and sweeps.
Everything in here except this README is gitignored. The layout is
deterministic and meant to be consumed by `klip plot`, `klip report`,
and `klip artifacts ls`.

## Layout

```
artifacts/
├── README.md                                  (this file, checked in)
├── experiment_plans/
│   └── pXX_<phase>.yaml        # generated sweep plans from `pixi run experiment-plans`
├── runs/
│   └── <experiment_name>/
│       └── <algo_kind>/
│           └── <env_id>/
│               └── seed=<n>/
│                   └── <UTC-timestamp>__<git_short_sha>/
│                       ├── snapshot.json      # resolved ExperimentConfig (deterministic)
│                       ├── metadata.json      # git, lockfile, host, exit status
│                       ├── config.input.yaml  # the YAML the user passed, when applicable
│                       ├── stdout.log         # per-iteration metric JSONL
│                       ├── logs/
│                       │   ├── console.log    # plain app logs, matching terminal style
│                       │   └── events.jsonl   # structured app lifecycle events
│                       ├── metrics/
│                       │   └── train.parquet  # one row per logged iteration
│                       └── checkpoints/
│                           ├── policy_step_<n>.pt
│                           └── final.pt
├── sweeps/
│   └── <UTC-timestamp>__<sweep-name>/
│       ├── manifest.json    # immutable: jobs + slots + concurrency
│       ├── results.json     # per-job exit codes and run dirs
│       └── logs/
│           ├── console.log
│           ├── events.jsonl
│           └── <label>__seed<n>.log
└── reports/
    └── <YYYY-MM-DD>/
        ├── learning_curves.pdf
        ├── ...
        └── report.md
```

## Reproducing a run

`snapshot.json` is a fully resolved `ExperimentConfig`. `metadata.json`
carries the git commit and pixi.lock sha256 used at training time.
Re-execute with:

```bash
pixi run train --from-snapshot artifacts/runs/<...>/snapshot.json --seed 0
```

(The seed override is required because `snapshot.json` is the
configuration, not the per-seed instance.)

## Generating experiment plans

Generate the staged local sweep plan YAMLs after refreshing snapshots:

```bash
pixi run materialize
pixi run experiment-plans
```

The generated `artifacts/experiment_plans/*.yaml` files are local artifacts.
They can be edited for host-specific GPU slots before launching `pixi run sweep`.

## Logs

Training writes human-readable lifecycle logs to `logs/console.log` and structured
JSONL lifecycle events to `logs/events.jsonl`. Iteration metrics are still written
to `stdout.log` for backward compatibility and to `metrics/train.parquet` for
analysis. When W&B logging is enabled, the run artifact includes the plain log,
structured JSONL log, metric JSONL, parquet metrics, metadata, snapshot, and final
checkpoint when present.

## Pulling Modal artifacts

Remote runs use the same layout inside the `klip-ppo-artifacts` Modal Volume.
Download them into this local directory with:

```bash
pixi run modal-pull --all
```
