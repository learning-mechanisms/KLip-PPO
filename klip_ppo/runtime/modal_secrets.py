"""Publish local WandB environment variables as a Modal secret."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

REQUIRED_WANDB_SECRET_ENV_VARS = ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT")
OPTIONAL_WANDB_SECRET_ENV_VARS = ("WANDB_MODE",)
DEFAULT_WANDB_SECRET_NAME = "wandb"


def main(argv: Sequence[str] | None = None) -> int:
    """Publish W&B settings from the current environment to Modal."""
    parser = argparse.ArgumentParser(
        prog="klip_ppo.runtime.modal_secrets",
        description="Create or update the Modal W&B secret from env vars.",
    )
    parser.add_argument(
        "--secret-name",
        default=os.environ.get("KLIP_MODAL_WANDB_SECRET", DEFAULT_WANDB_SECRET_NAME),
        help="Modal secret name to create or overwrite.",
    )
    parser.add_argument(
        "--modal-environment",
        default=os.environ.get("MODAL_ENVIRONMENT"),
        help="Optional Modal environment name.",
    )
    args = parser.parse_args(argv)

    values = _read_wandb_secret_env()
    modal = shutil.which("modal")
    if modal is None:
        raise RuntimeError("Could not find the `modal` CLI on PATH.")

    secret_file = _write_temp_secret_json(values)
    command = [modal, "secret", "create", "--force"]
    if args.modal_environment:
        command.extend(["--env", args.modal_environment])
    command.extend(["--from-json", str(secret_file), args.secret_name])

    try:
        subprocess.run(command, check=True)
    finally:
        secret_file.unlink(missing_ok=True)

    keys = ", ".join(values)
    print(f"Published Modal secret {args.secret_name!r} with keys: {keys}.")
    return 0


def _read_wandb_secret_env() -> dict[str, str]:
    values = {name: os.environ.get(name) for name in REQUIRED_WANDB_SECRET_ENV_VARS}
    missing = [name for name, value in values.items() if not value]
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variables: {missing_names}. "
            "Set them in .env or the shell before running this task."
        )

    payload = {name: value for name, value in values.items() if value is not None}
    for name in OPTIONAL_WANDB_SECRET_ENV_VARS:
        if value := os.environ.get(name):
            payload[name] = value
    return payload


def _write_temp_secret_json(values: dict[str, str]) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="klip-ppo-modal-secret-",
        suffix=".json",
        delete=False,
    ) as handle:
        path = Path(handle.name)
        path.chmod(0o600)
        json.dump(values, handle)
        handle.write("\n")
    return path


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
