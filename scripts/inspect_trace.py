#!/usr/bin/env python3
"""Print the latest local agent trace as a parent/child span tree."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(
            os.getenv(
                "TRACE_FILE",
                "logs/crossword-agent-traces.jsonl",
            )
        ),
    )
    parser.add_argument("--trace-id", help="Trace ID to inspect; defaults to the latest")
    args = parser.parse_args()

    if not args.file.is_file():
        raise SystemExit(f"Trace file does not exist: {args.file}")
    spans = [
        json.loads(line)
        for line in args.file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not spans:
        raise SystemExit(f"Trace file is empty: {args.file}")

    trace_id = args.trace_id or max(spans, key=lambda span: span["end_time"])["trace_id"]
    selected = [span for span in spans if span["trace_id"] == trace_id]
    if not selected:
        raise SystemExit(f"Trace ID not found: {trace_id}")

    by_id = {span["span_id"]: span for span in selected}
    children = defaultdict(list)
    roots = []
    for span in selected:
        parent_id = span.get("parent_span_id")
        if parent_id and parent_id in by_id:
            children[parent_id].append(span)
        else:
            roots.append(span)
    for siblings in children.values():
        siblings.sort(key=lambda span: span["start_time"])
    roots.sort(key=lambda span: span["start_time"])

    print(f"Trace {trace_id}")
    for index, root in enumerate(roots):
        _print_span(
            root,
            children=children,
            prefix="",
            last=index == len(roots) - 1,
        )


def _print_span(
    span,
    *,
    children,
    prefix: str,
    last: bool,
) -> None:
    connector = "└─" if last else "├─"
    status = span["status"]["code"]
    detail = _span_detail(span)
    print(
        f"{prefix}{connector} {span['name']} "
        f"[{status}] {span['duration_ms']:.1f}ms{detail}"
    )
    descendants = children.get(span["span_id"], [])
    child_prefix = f"{prefix}{'   ' if last else '│  '}"
    for index, child in enumerate(descendants):
        _print_span(
            child,
            children=children,
            prefix=child_prefix,
            last=index == len(descendants) - 1,
        )


def _span_detail(span) -> str:
    attributes = span.get("attributes", {})
    details = [
        attributes.get("crossword.entry.id"),
        attributes.get("gen_ai.request.model"),
        attributes.get("agent.status"),
    ]
    rendered = [str(detail) for detail in details if detail]
    return f" · {' · '.join(rendered)}" if rendered else ""


if __name__ == "__main__":
    main()
