#!/usr/bin/env python3
"""Run the same benchmark sample for each configured Nebius model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from crossword_agent.benchmark import select_benchmark_records, write_benchmark_collection
from crossword_agent.config import load_local_env
from crossword_agent.evaluation import (
    PuzzleEvaluation,
    evaluate_records,
    load_normalized_dataset,
    summarize,
)
from crossword_agent.logging_config import configure_logging
from crossword_agent.providers.nebius import NEBIUS_MODEL_OPTIONS, NebiusCandidateProvider
from crossword_agent.runtime import create_lexicon


def main() -> None:
    load_local_env()
    configure_logging()
    asyncio.run(_run(_parse_args()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark every configured Nebius model on one reproducible puzzle sample.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            os.getenv("CROSSWORD_DATASET_PATH", "data/crosswordbench/demo-7x7.json")
        ),
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--status-interval",
        type=float,
        default=15.0,
        help="Seconds between heartbeat status lines while model calls are running",
    )
    parser.add_argument(
        "--models",
        default=",".join(NEBIUS_MODEL_OPTIONS),
        help="Comma-separated Nebius model IDs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmark/models.json"),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing collection and rerun every model",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset does not exist: {args.dataset}")
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    if args.status_interval <= 0:
        raise SystemExit("--status-interval must be positive")

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise SystemExit("At least one model ID is required")
    unsupported = sorted(set(models) - set(NEBIUS_MODEL_OPTIONS))
    if unsupported:
        raise SystemExit(
            "Unsupported benchmark model(s): "
            f"{', '.join(unsupported)}. Choose from {', '.join(NEBIUS_MODEL_OPTIONS)}."
        )
    records = load_normalized_dataset(args.dataset)
    selected = select_benchmark_records(records, count=args.count, seed=args.seed)
    lexicon = create_lexicon()
    results_by_model: dict[str, list[PuzzleEvaluation]] = {}
    print(
        f"[setup] dataset={args.dataset} selected={len(selected)} "
        f"models={len(models)} concurrency={args.concurrency} "
        f"status_interval={args.status_interval:.0f}s",
        flush=True,
    )
    if args.output.is_file():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            requested_models = set(models)
            for model, item in existing.get("models", {}).items():
                if model not in requested_models or args.fresh:
                    continue
                results_by_model[model] = [
                    PuzzleEvaluation.model_validate(result)
                    for result in item.get("puzzles", [])
                ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            results_by_model = {}
    started = time.perf_counter()

    for model_index, model in enumerate(models, start=1):
        existing_results = list(results_by_model.get(model, ()))
        if len(existing_results) >= args.count:
            print(
                f"[skip {model_index}/{len(models)}] {model} already has "
                f"{args.count} completed puzzles",
                flush=True,
            )
            continue
        completed_ids = {result.puzzle_id for result in existing_results}
        pending = [record for record in selected if record.puzzle.id not in completed_ids]
        results = existing_results
        results_by_model[model] = results
        model_started = time.perf_counter()
        print(
            f"[model {model_index}/{len(models)}] starting {model} "
            f"({len(pending)} remaining, {len(results)}/{args.count} already complete)",
            flush=True,
        )

        async def heartbeat(
            *,
            current_model: str = model,
            current_results: list[PuzzleEvaluation] = results,
            current_model_started: float = model_started,
        ) -> None:
            while True:
                await asyncio.sleep(args.status_interval)
                elapsed = time.perf_counter() - current_model_started
                print(
                    f"[status] {current_model} completed={len(current_results)}/{args.count} "
                    f"elapsed={elapsed:.0f}s active~{args.concurrency}",
                    flush=True,
                )

        async def on_result(
            _index: int,
            result: PuzzleEvaluation,
            *,
            current_model: str = model,
            current_model_index: int = model_index,
            current_results: list[PuzzleEvaluation] = results,
        ) -> None:
            current_results.append(result)
            results_by_model[current_model] = current_results
            write_benchmark_collection(
                args.output,
                results_by_model=results_by_model,
                dataset=str(args.dataset.resolve()),
                provider="nebius",
                seed=args.seed,
                requested_puzzles=args.count,
            )
            print(
                f"[{current_model_index}/{len(models)}] {current_model} "
                f"{len(current_results):02d}/{args.count:02d} "
                f"{'SOLVED' if result.full_puzzle_solved else 'UNSOLVED'} "
                f"words={result.word_accuracy:.1%} "
                f"calls={result.model_calls} "
                f"elapsed={result.elapsed_ms / 1000:.1f}s",
                flush=True,
            )

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            await evaluate_records(
                pending,
                provider_factory=lambda model=model: NebiusCandidateProvider(
                    model_id=model,
                    api_key=os.getenv("NEBIUS_API_KEY"),
                ),
                mode="full",
                lexicon=lexicon,
                max_concurrency=args.concurrency,
                on_result=on_result,
            )
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

        summary = summarize(results)
        solved = sum(result.full_puzzle_solved for result in results)
        print(
            f"[model {model_index}/{len(models)}] finished {model} "
            f"solved={solved}/{len(results)} "
            f"word_accuracy={summary.word_accuracy:.1%} "
            f"mean_elapsed={summary.average_elapsed_ms / 1000:.1f}s "
            f"wall={time.perf_counter() - model_started:.1f}s",
            flush=True,
        )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "models": models,
                "puzzles_per_model": args.count,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
