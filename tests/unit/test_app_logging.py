"""Application logging setup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from klip_ppo.utils.log import configure_logging, get_logger, shutdown_logging


def test_configure_logging_writes_plain_and_json_logs(tmp_path: Path) -> None:
    plain = tmp_path / "console.log"
    events = tmp_path / "events.jsonl"

    try:
        log = configure_logging(
            console=False,
            plain_log_file=plain,
            json_log_file=events,
        ).bind(run="smoke")
        log.info("hello", answer=42)
    finally:
        shutdown_logging()

    assert "hello" in plain.read_text()
    row = json.loads(events.read_text().splitlines()[0])
    assert row["event"] == "hello"
    assert row["answer"] == 42
    assert row["run"] == "smoke"
    assert row["level"] == "info"
    assert "timestamp" in row


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    try:
        configure_logging(console=False, json_log_file=first)
        configure_logging(console=False, json_log_file=second)
        get_logger("test").info("once")
    finally:
        shutdown_logging()

    assert not first.exists() or first.read_text() == ""
    assert len(second.read_text().splitlines()) == 1
    assert logging.getLogger().handlers == []
