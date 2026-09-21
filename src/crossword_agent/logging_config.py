from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_CONTEXT_FIELDS = (
    "trace_id",
    "span_id",
    "request_id",
    "run_id",
    "puzzle_id",
    "entry_id",
    "phase",
    "event_message",
    "provider",
    "attempt",
    "request_count",
    "method",
    "path",
    "status",
    "status_code",
    "duration_ms",
    "model_calls",
    "log_file",
    "trace_file",
)
_HANDLER_MARKER = "_crossword_agent_handler"


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _CONTEXT_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            exception_type, exception, _traceback = record.exc_info
            payload["exception"] = {
                "type": exception_type.__name__ if exception_type else "Exception",
                "message": str(exception) if exception else "",
                "stacktrace": self.formatException(record.exc_info),
            }
        return json.dumps(payload, ensure_ascii=False, default=str)


class ContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        context = " ".join(
            f"{field}={getattr(record, field)}"
            for field in _CONTEXT_FIELDS
            if hasattr(record, field)
        )
        return f"{rendered} {context}" if context else rendered


class TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        from crossword_agent.tracing import current_trace_context

        context = current_trace_context()
        if context is not None:
            record.trace_id = context.trace_id
            record.span_id = context.span_id
        return True


def configure_logging(
    *,
    log_file: Path | None = None,
    force: bool = False,
) -> Path | None:
    logger = logging.getLogger("crossword_agent")
    if not force and any(
        getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers
    ):
        configured_file = os.getenv("LOG_FILE", "logs/crossword-agent.jsonl")
        return Path(configured_file).resolve()

    _remove_configured_handlers(logger)
    level = _log_level(os.getenv("LOG_LEVEL", "INFO"))
    logger.setLevel(level)
    logger.propagate = False

    console = logging.StreamHandler()
    setattr(console, _HANDLER_MARKER, True)
    console.setLevel(level)
    console.addFilter(TraceContextFilter())
    console.setFormatter(
        ContextFormatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(console)

    resolved_file = (log_file or Path(os.getenv("LOG_FILE", "logs/crossword-agent.jsonl")))
    resolved_file = resolved_file.resolve()
    try:
        resolved_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            resolved_file,
            maxBytes=int(os.getenv("LOG_MAX_BYTES", "5000000")),
            backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            encoding="utf-8",
            delay=True,
        )
    except (OSError, ValueError):
        logger.exception(
            "Unable to configure local file logging",
            extra={"log_file": str(resolved_file)},
        )
        return None

    setattr(file_handler, _HANDLER_MARKER, True)
    file_handler.setLevel(level)
    file_handler.addFilter(TraceContextFilter())
    file_handler.setFormatter(JsonLineFormatter())
    logger.addHandler(file_handler)
    logger.info("Logging configured", extra={"log_file": str(resolved_file)})
    return resolved_file


def _remove_configured_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if not getattr(handler, _HANDLER_MARKER, False):
            continue
        logger.removeHandler(handler)
        handler.close()


def _log_level(value: str) -> int:
    return getattr(logging, value.upper(), logging.INFO)
