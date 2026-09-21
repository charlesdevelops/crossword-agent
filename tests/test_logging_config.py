from __future__ import annotations

import json
import logging
from io import StringIO

from crossword_agent.logging_config import JsonLineFormatter, configure_logging


def test_json_formatter_includes_context_and_stacktrace() -> None:
    output = StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonLineFormatter())
    logger = logging.getLogger("formatter-test")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        raise ValueError("broken provider response")
    except ValueError:
        logger.exception(
            "Candidate generation failed",
            extra={"run_id": "run-123", "phase": "generate"},
        )

    payload = json.loads(output.getvalue())
    assert payload["level"] == "ERROR"
    assert payload["run_id"] == "run-123"
    assert payload["phase"] == "generate"
    assert payload["exception"]["type"] == "ValueError"
    assert "broken provider response" in payload["exception"]["stacktrace"]


def test_configure_logging_writes_json_lines(tmp_path) -> None:
    log_file = tmp_path / "crossword-agent.jsonl"
    configured = configure_logging(log_file=log_file, force=True)
    logger = logging.getLogger("crossword_agent.test")

    logger.info(
        "Run observed",
        extra={"run_id": "run-456", "status": "SOLVED"},
    )
    for handler in logging.getLogger("crossword_agent").handlers:
        handler.flush()

    payloads = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert configured == log_file
    assert payloads[-1]["message"] == "Run observed"
    assert payloads[-1]["run_id"] == "run-456"
    assert payloads[-1]["status"] == "SOLVED"

    configure_logging(force=True)
