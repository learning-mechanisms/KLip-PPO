"""Stage, validate, size, archive, and unpack-test the artifact."""

from __future__ import annotations

import sys

from submission import archive, size, stage, unpack, validate

COMMANDS = {
    "stage": stage.stage,
    "validate": validate.validate,
    "size": size.report,
    "zip": archive.write,
    "unpack": unpack.unpack_test,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python -m submission {{{'|'.join(COMMANDS)}}}")
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
