# analysis

Reproducible paper figures from frozen data.

## Layout

- `datasets/` — frozen datasets (`baselines.parquet`, `sweeps.parquet`), their `schema`, and the run lock.
- `export/` — online stage: pull the pinned runs and write `datasets/` (needs credentials).
- `render/` — offline stage: `datasets/` to figures and tables, byte-deterministic.

## Commands

    pixi run figures       # datasets/ -> paper/figures, verified against figures.sha256
    pixi run export-data   # refresh datasets/ from the pinned runs (online)

`figures.sha256` pins the expected output; the build is checked against it in the tests.
