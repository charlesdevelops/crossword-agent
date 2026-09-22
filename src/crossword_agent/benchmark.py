from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from crossword_agent.evaluation import PuzzleEvaluation, summarize
from crossword_agent.models import PuzzleRecord


def select_benchmark_records(
    records: Sequence[PuzzleRecord],
    *,
    count: int = 20,
    seed: int = 20260920,
) -> tuple[PuzzleRecord, ...]:
    if count < 1:
        raise ValueError("Benchmark puzzle count must be positive")
    if len(records) < count:
        raise ValueError(
            f"Benchmark requested {count} puzzles but the dataset contains {len(records)}"
        )
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return tuple(selected[:count])


def write_benchmark_dashboard(
    path: Path,
    *,
    results: Sequence[PuzzleEvaluation],
    dataset: str,
    provider: str,
    model: str,
    seed: int,
    requested_puzzles: int,
) -> None:
    if not results:
        raise ValueError("Cannot write an empty benchmark dashboard")
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    payload = {
        "metadata": {
            "generated_at": generated_at,
            "dataset": dataset,
            "provider": provider,
            "model": model,
            "seed": seed,
            "requested_puzzles": requested_puzzles,
            "completed_puzzles": len(results),
        },
        "summary": summary.model_dump(mode="json"),
        "puzzles": [result.model_dump(mode="json") for result in results],
    }
    raw_path = path.with_suffix(".json")
    raw_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    path.write_text(
        _render_dashboard(
            results=results,
            dataset=dataset,
            provider=provider,
            model=model,
            seed=seed,
            requested_puzzles=requested_puzzles,
            generated_at=generated_at,
            raw_name=raw_path.name,
        ),
        encoding="utf-8",
    )


def write_benchmark_collection(
    path: Path,
    *,
    results_by_model: Mapping[str, Sequence[PuzzleEvaluation]],
    dataset: str,
    provider: str,
    seed: int,
    requested_puzzles: int,
) -> None:
    if not results_by_model:
        raise ValueError("Cannot write an empty benchmark collection")
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    models = list(results_by_model)
    payload = {
        "metadata": {
            "generated_at": generated_at,
            "dataset": dataset,
            "provider": provider,
            "seed": seed,
            "requested_puzzles": requested_puzzles,
            "completed_models": len(models),
            "models": models,
        },
        "models": {
            model: {
                "metadata": {
                    "model": model,
                    "completed_puzzles": len(results),
                },
                "summary": summarize(results).model_dump(mode="json"),
                "puzzles": [result.model_dump(mode="json") for result in results],
            }
            for model, results in results_by_model.items()
            if results
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_dashboard(
    *,
    results: Sequence[PuzzleEvaluation],
    dataset: str,
    provider: str,
    model: str,
    seed: int,
    requested_puzzles: int,
    generated_at: str,
    raw_name: str,
) -> str:
    summary = summarize(results)
    completed = len(results)
    progress = completed / requested_puzzles
    recovery = _format_percent(summary.conflict_recovery_rate)
    rows = "\n".join(
        _puzzle_row(index, result) for index, result in enumerate(results, start=1)
    )
    escaped_raw_name = escape(raw_name)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crossword Agent Benchmark</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      --ink: #10243a;
      --muted: #687684;
      --surface: #ffffff;
      --background: #f5f6f1;
      --line: #dfe3da;
      --accent: #dfff52;
      --good: #52704f;
      --bad: #a5443c;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--background); color: var(--ink); }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 28px 24px 48px; }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 20px;
    }}
    h1 {{ margin: 0; font-size: clamp(2.2rem, 5vw, 4.5rem); letter-spacing: -.055em; }}
    h2 {{ margin: 0 0 12px; }}
    .meta {{ color: var(--muted); text-align: right; font-size: .9rem; line-height: 1.6; }}
    .progress {{
      height: 8px;
      overflow: hidden;
      background: #e7eadf;
      border-radius: 4px;
      margin-bottom: 22px;
    }}
    .progress > span {{
      display: block;
      width: {progress:.2%};
      height: 100%;
      background: var(--accent);
    }}
    .groups {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .group, .table-card, .notes {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 10px 30px #10243a0a;
    }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .metric {{ min-height: 92px; background: #f2f4ee; border-radius: 6px; padding: 12px; }}
    .metric span {{ display: block; color: var(--muted); font-size: .78rem; font-weight: 750; }}
    .metric strong {{
      display: block;
      margin-top: 8px;
      font-size: 1.55rem;
      letter-spacing: -.03em;
    }}
    .metric small {{ display: block; margin-top: 4px; color: var(--muted); }}
    .table-card {{ margin-top: 14px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
    th, td {{
      padding: 9px 8px;
      border-bottom: 1px solid #e8ebe3;
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-size: .7rem;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tbody tr:hover {{ background: #f7f8f4; }}
    .solved {{ color: var(--good); font-weight: 800; }}
    .unsolved {{ color: var(--bad); font-weight: 800; }}
    .notes {{ margin-top: 14px; color: var(--muted); font-size: .88rem; line-height: 1.6; }}
    .notes ul {{ margin-bottom: 0; }}
    a {{ color: var(--ink); }}
    @media (max-width: 980px) {{
      header {{ align-items: start; flex-direction: column; }}
      .meta {{ text-align: left; }}
      .groups {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Agent Benchmark</h1>
    <div class="meta">
      <div>{escape(provider)} · {escape(model)}</div>
      <div>{completed}/{requested_puzzles} puzzles · seed {seed}</div>
      <div>{escape(generated_at)}</div>
    </div>
  </header>
  <div class="progress" aria-label="Benchmark progress"><span></span></div>

  <section class="groups">
    <article class="group">
      <h2>Quality</h2>
      <div class="metrics">
        {_metric("Puzzle solve rate", f"{summary.full_puzzle_solve_rate:.1%}")}
        {_metric("Word accuracy", f"{summary.word_accuracy:.1%}")}
        {_metric("Letter accuracy", f"{summary.letter_accuracy:.1%}")}
        {_metric("Intersection consistency", f"{summary.intersection_consistency:.1%}")}
      </div>
    </article>
    <article class="group">
      <h2>Agent behaviour</h2>
      <div class="metrics">
        {_metric("Conflict recovery", recovery, f"{summary.recovery_opportunities} opportunities")}
        {_metric("Revisions / puzzle", f"{summary.average_candidate_replacements:.2f}")}
        {_metric("Model calls / puzzle", f"{summary.average_model_calls:.2f}")}
        {_metric("Constraint violations", str(summary.constraint_violations), "total")}
      </div>
    </article>
    <article class="group">
      <h2>Candidate recall</h2>
      <div class="metrics">
        {_metric("Initial recall @1", f"{summary.initial_candidate_recall_at_1:.1%}")}
        {_metric("Initial recall @5", f"{summary.initial_candidate_recall_at_5:.1%}")}
        {_metric("Final recall @5", f"{summary.final_candidate_recall_at_5:.1%}")}
        {_metric("Final recall @10", f"{summary.final_candidate_recall_at_10:.1%}")}
        {_metric("Final candidate MRR", f"{summary.final_candidate_mrr:.3f}")}
        {_metric(
            "Oracle solve rate",
            f"{summary.final_oracle_solve_rate:.1%}",
            "gold exists in every final domain",
        )}
      </div>
    </article>
    <article class="group">
      <h2>Efficiency</h2>
      <div class="metrics">
        {_metric("Tokens / puzzle", f"{summary.average_total_tokens:,.0f}")}
        {_metric("Mean latency", _format_duration(summary.average_elapsed_ms), "end-to-end")}
      </div>
    </article>
  </section>

  <section class="table-card">
    <h2>Per-puzzle results</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Puzzle</th><th>Result</th><th>Words</th><th>Letters</th>
          <th>Crossings</th><th>Candidates @5</th><th>Oracle</th><th>Recovered</th>
          <th>Revisions</th><th>Calls</th><th>Tokens</th><th>Latency</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>

  <section class="notes">
    <strong>Methodology</strong>
    <ul>
      <li>Puzzle solve rate requires every reference word to match exactly.</li>
      <li>
        Intersection consistency counts a crossing only when both words are present and agree.
      </li>
      <li>
        Conflict recovery measures initially wrong word decisions corrected by the final assignment.
      </li>
      <li>
        Candidate recall measures whether the reference answer entered the ranked domains; oracle
        solve rate requires every reference answer to be present somewhere in its final domain.
      </li>
      <li>Revisions are candidate replacements; model calls are provider invocations.</li>
    </ul>
    <p>Dataset: <code>{escape(dataset)}</code> · <a href="{escaped_raw_name}">Raw JSON</a></p>
  </section>
</main>
</body>
</html>
"""


def _metric(label: str, value: str, detail: str = "") -> str:
    rendered_detail = f"<small>{escape(detail)}</small>" if detail else ""
    return (
        f'<div class="metric"><span>{escape(label)}</span>'
        f"<strong>{escape(value)}</strong>{rendered_detail}</div>"
    )


def _puzzle_row(index: int, result: PuzzleEvaluation) -> str:
    result_label = "Solved" if result.full_puzzle_solved else "Unsolved"
    result_class = "solved" if result.full_puzzle_solved else "unsolved"
    recovered = (
        f"{result.recovered_conflicts}/{result.recovery_opportunities}"
        if result.recovery_opportunities
        else "n/a"
    )
    tokens = result.input_tokens + result.output_tokens
    return (
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{escape(result.puzzle_id)}</td>"
        f'<td class="{result_class}">{result_label}</td>'
        f"<td>{result.word_accuracy:.1%}</td>"
        f"<td>{result.letter_accuracy:.1%}</td>"
        f"<td>{result.intersection_consistency:.1%}</td>"
        f"<td>{result.final_candidate_recall_at_5:.1%}</td>"
        f"<td>{'Yes' if result.final_oracle_solvable else 'No'}</td>"
        f"<td>{recovered}</td>"
        f"<td>{result.candidate_replacements}</td>"
        f"<td>{result.model_calls}</td>"
        f"<td>{tokens:,}</td>"
        f"<td>{_format_duration(result.elapsed_ms)}</td>"
        "</tr>"
    )


def _format_percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def _format_duration(value_ms: float) -> str:
    return f"{value_ms / 1000:.1f}s"
