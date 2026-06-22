# Artifact

A self-contained bundle that rebuilds every paper figure from frozen data.

```bash
pixi run python -m submission validate   # offline build matches checksums
pixi run python -m submission stage      # assemble dist/artifact
pixi run python -m submission size       # report and bound size
pixi run python -m submission zip        # byte-stable dist/artifact.zip
pixi run python -m submission unpack     # extract clean and rebuild
```

The staged `run.sh` reproduces the figures from `analysis/datasets/` with no network access.
