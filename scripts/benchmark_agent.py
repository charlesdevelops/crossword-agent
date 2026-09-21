#!/usr/bin/env python3
"""Run a reproducible multi-puzzle agent benchmark and write an HTML dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from crossword_agent.benchmark import (
    select_benchmark_records,
    write_benchmark_dashboard,
)
from crossword_agent.config import load_local_env
from crossword_agent.evaluation import PuzzleEvaluation, evaluate_records, load_normalized_dataset
from crossword_agent.logging_config import configure_logging
from crossword_agent.providers.base import CandidateProvider
from crossword_agent.providers.bedrock import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_BEDROCK_REGION,
    BedrockCandidateProvider,
)
from crossword_agent.providers.nebius import DEFAULT_NEBIUS_MODEL, NebiusCandidateProvider
from crossword_agent.runtime import create_clue_index, create_lexicon


def main() -> None:
    load_local_env()
    configure_logging()
    args = _parse_args()
    asyncio.run(_run(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the full crossword agent and write a self-contained HTML report.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            os.getenv(
                "CROSSWORD_DATASET_PATH",
                "data/crosswordbench/demo-7x7.json",
            )
        ),
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of independent puzzles to solve concurrently",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--provider",
        choices=("bedrock", "nebius"),
        default=os.getenv("LLM_PROVIDER", "bedrock"),
    )
    parser.add_argument("--model", help="Override the configured provider model ID")
    parser.add_argument(
        "--region",
        default=os.getenv("BEDROCK_REGION", DEFAULT_BEDROCK_REGION),
        help="Bedrock region; ignored for Nebius",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmark/dashboard.html"),
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise SystemExit(
            f"Dataset does not exist: {args.dataset}. "
            "Run scripts/prepare_crosswordbench.py first."
        )
    records = load_normalized_dataset(args.dataset)
    selected = select_benchmark_records(
        records,
        count=args.count,
        seed=args.seed,
    )
    model = _resolved_model(args.provider, args.model)
    provider_factory = _provider_factory(
        provider=args.provider,
        model=model,
        region=args.region,
    )
    lexicon = create_lexicon()
    clue_index = create_clue_index()
    results: list[PuzzleEvaluation] = []
    completed_count = 0
    benchmark_started = time.perf_counter()

    print(
        f"Benchmarking {len(selected)} puzzles with {args.provider}/{model} "
        f"({args.concurrency} concurrent puzzles)",
        flush=True,
    )

    async def on_result(_index: int, result: PuzzleEvaluation) -> None:
        nonlocal completed_count
        completed_count += 1
        results.append(result)
        write_benchmark_dashboard(
            args.output,
            results=results,
            dataset=str(args.dataset.resolve()),
            provider=args.provider,
            model=model,
            seed=args.seed,
            requested_puzzles=args.count,
        )
        print(
            (
                f"[{completed_count:02d}/{args.count:02d}] {result.puzzle_id} "
                f"{'SOLVED' if result.full_puzzle_solved else 'UNSOLVED'} "
                f"words={result.word_accuracy:.1%} "
                f"candidates@5={result.final_candidate_recall_at_5:.1%} "
                f"oracle={'yes' if result.final_oracle_solvable else 'no'} "
                f"calls={result.model_calls} "
                f"elapsed={result.elapsed_ms / 1000:.1f}s"
            ),
            flush=True,
        )

    await evaluate_records(
        selected,
        provider_factory=provider_factory,
        mode="full",
        lexicon=lexicon,
        clue_index=clue_index,
        max_concurrency=args.concurrency,
        on_result=on_result,
    )

    print(
        json.dumps(
            {
                "dashboard": str(args.output),
                "raw": str(args.output.with_suffix(".json")),
                "puzzles": len(results),
                "elapsed_seconds": round(time.perf_counter() - benchmark_started, 1),
            },
            indent=2,
        )
    )


def _resolved_model(provider: str, override: str | None) -> str:
    if override:
        return override
    if provider == "bedrock":
        return os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL)
    return os.getenv("NEBIUS_MODEL_ID", DEFAULT_NEBIUS_MODEL)


def _provider_factory(
    *,
    provider: str,
    model: str,
    region: str,
):
    def create() -> CandidateProvider:
        if provider == "bedrock":
            return BedrockCandidateProvider(
                model_id=model,
                region_name=region,
            )
        return NebiusCandidateProvider(
            model_id=model,
            api_key=os.getenv("NEBIUS_API_KEY"),
        )

    return create


if __name__ == "__main__":
    main()
