from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import secrets
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_TRACEPARENT = re.compile(
    r"^(?P<version>[0-9a-f]{2})-"
    r"(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<span_id>[0-9a-f]{16})-"
    r"(?P<flags>[0-9a-f]{2})$"
)
_current_span: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "crossword_agent_current_span",
    default=None,
)
_exporter: JsonLineSpanExporter | None = None
_trace_file: Path | None = None


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str
    sampled: bool = True


class JsonLineSpanExporter:
    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int,
        backup_count: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger = logging.Logger("crossword_agent.local_trace_exporter", logging.INFO)
        self._logger.propagate = False
        self._logger.addHandler(self._handler)

    def export(self, payload: dict[str, Any]) -> None:
        self._logger.info(json.dumps(payload, ensure_ascii=False, default=str))

    def flush(self) -> None:
        self._handler.flush()

    def shutdown(self) -> None:
        self._handler.flush()
        self._handler.close()


class Span:
    def __init__(
        self,
        name: str,
        *,
        kind: str,
        attributes: Mapping[str, Any] | None,
        parent: TraceContext | None,
    ) -> None:
        active = _current_span.get()
        resolved_parent = parent or (active.context if active else None)
        self.name = name
        self.kind = kind
        self.context = TraceContext(
            trace_id=resolved_parent.trace_id if resolved_parent else secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            sampled=resolved_parent.sampled if resolved_parent else True,
        )
        self.parent_span_id = resolved_parent.span_id if resolved_parent else None
        self.attributes = {
            key: _json_value(value) for key, value in (attributes or {}).items()
        }
        self.events: list[dict[str, Any]] = []
        self.status = "UNSET"
        self.status_description: str | None = None
        self.exception: dict[str, str] | None = None
        self._started_ns = time.time_ns()
        self._token: contextvars.Token[Span | None] | None = None

    def __enter__(self) -> Span:
        self._token = _current_span.set(self)
        return self

    def __exit__(self, exception_type, exception, exception_traceback) -> bool:
        if exception is not None:
            self.record_exception(exception)
            self.set_status("ERROR", str(exception))
        self.end()
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = _json_value(value)

    def set_status(self, status: str, description: str | None = None) -> None:
        self.status = status.upper()
        self.status_description = description

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "name": name,
                "timestamp": _timestamp(time.time_ns()),
                "attributes": {
                    key: _json_value(value)
                    for key, value in (attributes or {}).items()
                },
            }
        )

    def record_exception(self, exception: BaseException) -> None:
        self.exception = {
            "type": type(exception).__name__,
            "message": str(exception),
            "stacktrace": "".join(
                traceback.format_exception(
                    type(exception),
                    exception,
                    exception.__traceback__,
                )
            ),
        }

    def end(self) -> None:
        ended_ns = time.time_ns()
        if self._token is not None:
            _current_span.reset(self._token)
            self._token = None
        if not self.context.sampled or _exporter is None:
            return
        _exporter.export(
            {
                "trace_id": self.context.trace_id,
                "span_id": self.context.span_id,
                "parent_span_id": self.parent_span_id,
                "name": self.name,
                "kind": self.kind,
                "start_time": _timestamp(self._started_ns),
                "end_time": _timestamp(ended_ns),
                "duration_ms": round((ended_ns - self._started_ns) / 1_000_000, 3),
                "status": {
                    "code": self.status,
                    "description": self.status_description,
                },
                "attributes": self.attributes,
                "events": self.events,
                "exception": self.exception,
                "resource": {
                    "service.name": os.getenv(
                        "OTEL_SERVICE_NAME",
                        "crossword-agent",
                    ),
                    "deployment.environment": os.getenv(
                        "DEPLOYMENT_ENVIRONMENT",
                        "local",
                    ),
                },
            }
        )


def configure_tracing(
    *,
    trace_file: Path | None = None,
    force: bool = False,
) -> Path | None:
    global _exporter, _trace_file
    if os.getenv("TRACE_ENABLED", "true").lower() in {"0", "false", "no"}:
        return None
    if _exporter is not None and not force:
        return _trace_file
    if _exporter is not None:
        _exporter.shutdown()
    resolved = (
        trace_file
        or Path(os.getenv("TRACE_FILE", "logs/crossword-agent-traces.jsonl"))
    ).resolve()
    try:
        _exporter = JsonLineSpanExporter(
            resolved,
            max_bytes=int(os.getenv("TRACE_MAX_BYTES", "10000000")),
            backup_count=int(os.getenv("TRACE_BACKUP_COUNT", "5")),
        )
    except (OSError, ValueError):
        logging.getLogger(__name__).exception(
            "Unable to configure local span tracing",
            extra={"trace_file": str(resolved)},
        )
        _exporter = None
        _trace_file = None
        return None
    _trace_file = resolved
    return resolved


def start_span(
    name: str,
    *,
    kind: str = "INTERNAL",
    attributes: Mapping[str, Any] | None = None,
    parent: TraceContext | None = None,
) -> Span:
    return Span(
        name,
        kind=kind.upper(),
        attributes=attributes,
        parent=parent,
    )


def current_trace_context() -> TraceContext | None:
    span = _current_span.get()
    return span.context if span else None


def extract_traceparent(headers: Mapping[str, str]) -> TraceContext | None:
    value = headers.get("traceparent", "").lower()
    match = _TRACEPARENT.fullmatch(value)
    if not match:
        return None
    trace_id = match.group("trace_id")
    span_id = match.group("span_id")
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        sampled=bool(int(match.group("flags"), 16) & 1),
    )


def format_traceparent(context: TraceContext) -> str:
    flags = "01" if context.sampled else "00"
    return f"00-{context.trace_id}-{context.span_id}-{flags}"


def force_flush_traces() -> None:
    if _exporter is not None:
        _exporter.flush()


def _timestamp(timestamp_ns: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ns / 1_000_000_000, UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]
    return str(value)
