from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from crossword_agent.agent import AgentSettings, solve_puzzle
from crossword_agent.constraints import (
    candidate_fits,
    count_constraint_violations,
    search_assignments,
    validate_domain,
)
from crossword_agent.lexicon import PatternLexicon
from crossword_agent.models import Candidate, ClueRequest, PuzzleRecord, SolveStatus
from crossword_agent.providers.base import CandidateProvider
from crossword_agent.puzzles import load_normalized_records
from crossword_agent.retrieval import ClueAnswerIndex

AblationMode = Literal["single_shot", "top1_constraints", "constraint_search", "full"]


class PuzzleEvaluation(BaseModel):
    puzzle_id: str
    mode: AblationMode
    letter_accuracy: float
    word_accuracy: float
    full_puzzle_solved: bool
    intersection_consistency: float
    constraint_violations: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    elapsed_ms: float
    estimated_cost_usd: float | None = None
    candidate_replacements: int
    recovery_opportunities: int
    recovered_conflicts: int
    conflict_recovery_rate: float | None
    initial_candidate_recall_at_1: float
    initial_candidate_recall_at_5: float
    initial_candidate_recall_at_10: float
    final_candidate_recall_at_1: float
    final_candidate_recall_at_5: float
    final_candidate_recall_at_10: float
    initial_candidate_mrr: float
    final_candidate_mrr: float
    initial_oracle_solvable: bool
    final_oracle_solvable: bool
    assignment: dict[str, str]
    status: SolveStatus | None = None


class EvaluationSummary(BaseModel):
    mode: AblationMode
    puzzles: int
    letter_accuracy: float
    word_accuracy: float
    full_puzzle_solve_rate: float
    intersection_consistency: float
    constraint_violations: int
    conflict_recovery_rate: float | None
    recovery_opportunities: int
    average_model_calls: float
    average_input_tokens: float
    average_output_tokens: float
    average_total_tokens: float
    average_latency_ms: float
    p50_latency_ms: float
    average_elapsed_ms: float
    p50_elapsed_ms: float
    average_cost_usd: float | None = None
    cost_per_successful_solve_usd: float | None = None
    average_candidate_replacements: float
    initial_candidate_recall_at_1: float
    initial_candidate_recall_at_5: float
    initial_candidate_recall_at_10: float
    final_candidate_recall_at_1: float
    final_candidate_recall_at_5: float
    final_candidate_recall_at_10: float
    initial_candidate_mrr: float
    final_candidate_mrr: float
    initial_oracle_solve_rate: float
    final_oracle_solve_rate: float


@dataclass(frozen=True)
class StudySplit:
    development: tuple[PuzzleRecord, ...]
    demo: tuple[PuzzleRecord, ...]
    heldout_7x7: tuple[PuzzleRecord, ...]
    heldout_14x14: tuple[PuzzleRecord, ...]
    ablation_subset: tuple[PuzzleRecord, ...]
    seed: int

    @property
    def heldout(self) -> tuple[PuzzleRecord, ...]:
        return self.heldout_7x7 + self.heldout_14x14


class ModelBakeoffResult(BaseModel):
    model: str
    word_accuracy: float
    full_puzzle_solve_rate: float
    average_cost_usd: float | None
    p50_latency_ms: float


@dataclass(frozen=True)
class ModeRunResult:
    assignment: dict[str, str]
    candidate_replacements: int
    initial_assignment: dict[str, str]
    elapsed_ms: float
    status: SolveStatus | None
    initial_candidate_pool: dict[str, tuple[str, ...]]
    final_candidate_pool: dict[str, tuple[str, ...]]


def load_normalized_dataset(path: Path) -> list[PuzzleRecord]:
    return load_normalized_records(path)


def deterministic_split(
    records: Sequence[PuzzleRecord],
    *,
    dev_count: int = 20,
    demo_count: int = 3,
    seed: int = 20260919,
) -> tuple[list[PuzzleRecord], list[PuzzleRecord], list[PuzzleRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return (
        shuffled[:dev_count],
        shuffled[dev_count : dev_count + demo_count],
        shuffled[dev_count + demo_count :],
    )


def build_study_split(
    records: Sequence[PuzzleRecord],
    *,
    demo_records: Sequence[PuzzleRecord] = (),
    seed: int = 20260919,
    dev_count: int = 20,
    demo_count: int = 3,
    heldout_7x7_count: int = 50,
    heldout_14x14_count: int = 20,
    ablation_count: int = 20,
) -> StudySplit:
    """Create the exact reproducible study split without exposing gold answers."""

    if len({record.puzzle.id for record in records}) != len(records):
        raise ValueError("Dataset puzzle IDs must be unique")
    explicit_demos = list(demo_records) or [
        record for record in records if record.split.lower() == "demo"
    ]
    if len(explicit_demos) < demo_count:
        raise ValueError(f"Study requires {demo_count} separate demo puzzles")
    demos = tuple(explicit_demos[:demo_count])
    demo_ids = {record.puzzle.id for record in demos}

    candidates = [record for record in records if record.puzzle.id not in demo_ids]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    explicit_development = [
        record
        for record in candidates
        if record.split.lower() in {"dev", "development"}
    ]
    if len(explicit_development) >= dev_count:
        development = explicit_development[:dev_count]
    else:
        explicit_ids = {record.puzzle.id for record in explicit_development}
        fill = [record for record in candidates if record.puzzle.id not in explicit_ids]
        development = [*explicit_development, *fill[: dev_count - len(explicit_development)]]
    if len(development) < dev_count:
        raise ValueError(f"Study requires at least {dev_count} development puzzles")

    development_ids = {record.puzzle.id for record in development}
    heldout_pool = [
        record for record in candidates if record.puzzle.id not in development_ids
    ]
    seven = [
        record
        for record in heldout_pool
        if record.puzzle.width == 7 and record.puzzle.height == 7
    ]
    fourteen = [
        record
        for record in heldout_pool
        if record.puzzle.width == 14 and record.puzzle.height == 14
    ]
    if len(seven) < heldout_7x7_count:
        raise ValueError(
            f"Study requires {heldout_7x7_count} held-out 7x7 puzzles; found {len(seven)}"
        )
    if len(fourteen) < heldout_14x14_count:
        raise ValueError(
            f"Study requires {heldout_14x14_count} held-out 14x14 puzzles; "
            f"found {len(fourteen)}"
        )
    heldout_7x7 = tuple(seven[:heldout_7x7_count])
    heldout_14x14 = tuple(fourteen[:heldout_14x14_count])
    heldout = [*heldout_7x7, *heldout_14x14]
    rng.shuffle(heldout)
    if len(heldout) < ablation_count:
        raise ValueError(f"Study requires {ablation_count} puzzles for ablations")
    return StudySplit(
        development=tuple(development),
        demo=demos,
        heldout_7x7=heldout_7x7,
        heldout_14x14=heldout_14x14,
        ablation_subset=tuple(heldout[:ablation_count]),
        seed=seed,
    )


async def evaluate_records(
    records: Sequence[PuzzleRecord],
    *,
    provider_factory: Callable[[], CandidateProvider],
    mode: AblationMode,
    lexicon: PatternLexicon | None = None,
    clue_index: ClueAnswerIndex | None = None,
    max_concurrency: int = 1,
    on_result: Callable[[int, PuzzleEvaluation], Awaitable[None]] | None = None,
) -> list[PuzzleEvaluation]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    semaphore = asyncio.Semaphore(max_concurrency)
    resolved_lexicon = lexicon or PatternLexicon.empty()
    resolved_clue_index = clue_index or ClueAnswerIndex.empty()

    async def evaluate_one(
        index: int,
        record: PuzzleRecord,
    ) -> tuple[int, PuzzleEvaluation]:
        async with semaphore:
            if record.solution is None:
                raise ValueError(f"Puzzle {record.puzzle.id} has no gold solution")
            provider = provider_factory()
            run = await _run_mode(
                record,
                provider=provider,
                mode=mode,
                lexicon=resolved_lexicon,
                clue_index=resolved_clue_index,
            )
            scores = score_assignment(record, run.assignment)
            recovery = score_conflict_recovery(
                record,
                initial_assignment=run.initial_assignment,
                final_assignment=run.assignment,
            )
            initial_recall = score_candidate_recall(
                record,
                run.initial_candidate_pool,
            )
            final_recall = score_candidate_recall(
                record,
                run.final_candidate_pool,
            )
            result = PuzzleEvaluation(
                puzzle_id=record.puzzle.id,
                mode=mode,
                assignment=run.assignment,
                candidate_replacements=run.candidate_replacements,
                model_calls=provider.usage.calls,
                input_tokens=provider.usage.input_tokens,
                output_tokens=provider.usage.output_tokens,
                latency_ms=provider.usage.latency_ms,
                elapsed_ms=run.elapsed_ms,
                estimated_cost_usd=provider.usage.estimated_cost_usd,
                status=run.status,
                initial_candidate_recall_at_1=initial_recall["recall_at_1"],
                initial_candidate_recall_at_5=initial_recall["recall_at_5"],
                initial_candidate_recall_at_10=initial_recall["recall_at_10"],
                final_candidate_recall_at_1=final_recall["recall_at_1"],
                final_candidate_recall_at_5=final_recall["recall_at_5"],
                final_candidate_recall_at_10=final_recall["recall_at_10"],
                initial_candidate_mrr=initial_recall["mrr"],
                final_candidate_mrr=final_recall["mrr"],
                initial_oracle_solvable=bool(initial_recall["oracle_solvable"]),
                final_oracle_solvable=bool(final_recall["oracle_solvable"]),
                **recovery,
                **scores,
            )
        if on_result is not None:
            await on_result(index, result)
        return index, result

    completed = await asyncio.gather(
        *(evaluate_one(index, record) for index, record in enumerate(records))
    )
    return [result for _index, result in sorted(completed)]


async def evaluate_study(
    split: StudySplit,
    *,
    provider_factory: Callable[[], CandidateProvider],
    lexicon: PatternLexicon | None = None,
    clue_index: ClueAnswerIndex | None = None,
) -> tuple[list[PuzzleEvaluation], dict[AblationMode, list[PuzzleEvaluation]]]:
    """Run the full agent on 70 held-out puzzles and four fixed-subset ablations."""

    resolved_lexicon = lexicon or PatternLexicon.empty()
    heldout_full = await evaluate_records(
        split.heldout,
        provider_factory=provider_factory,
        mode="full",
        lexicon=resolved_lexicon,
        clue_index=clue_index,
    )
    subset_ids = {record.puzzle.id for record in split.ablation_subset}
    ablations: dict[AblationMode, list[PuzzleEvaluation]] = {}
    for mode in ("single_shot", "top1_constraints", "constraint_search"):
        ablations[mode] = await evaluate_records(
            split.ablation_subset,
            provider_factory=provider_factory,
            mode=mode,
            lexicon=resolved_lexicon,
            clue_index=clue_index,
        )
    ablations["full"] = [
        result for result in heldout_full if result.puzzle_id in subset_ids
    ]
    return heldout_full, ablations


async def run_model_bakeoff(
    records: Sequence[PuzzleRecord],
    *,
    models: Sequence[str],
    provider_factory: Callable[[str], CandidateProvider],
    lexicon: PatternLexicon | None = None,
    clue_index: ClueAnswerIndex | None = None,
) -> list[ModelBakeoffResult]:
    """Evaluate candidate models and return them in the specified selection order."""

    results = []
    for model in models:
        evaluations = await evaluate_records(
            records,
            provider_factory=lambda model=model: provider_factory(model),
            mode="full",
            lexicon=lexicon or PatternLexicon.empty(),
            clue_index=clue_index,
        )
        summary = summarize(evaluations)
        results.append(
            ModelBakeoffResult(
                model=model,
                word_accuracy=summary.word_accuracy,
                full_puzzle_solve_rate=summary.full_puzzle_solve_rate,
                average_cost_usd=summary.average_cost_usd,
                p50_latency_ms=summary.p50_latency_ms,
            )
        )
    return sorted(
        results,
        key=lambda item: (
            -item.word_accuracy,
            -item.full_puzzle_solve_rate,
            item.average_cost_usd
            if item.average_cost_usd is not None
            else float("inf"),
            item.p50_latency_ms,
            item.model,
        ),
    )


async def _run_mode(
    record: PuzzleRecord,
    *,
    provider: CandidateProvider,
    mode: AblationMode,
    lexicon: PatternLexicon,
    clue_index: ClueAnswerIndex,
) -> ModeRunResult:
    started = time.perf_counter()
    puzzle = record.puzzle
    requests = [
        ClueRequest(
            entry_id=entry.id,
            clue=entry.clue,
            length=entry.length,
            pattern="?" * entry.length,
        )
        for entry in puzzle.entries
    ]
    if mode == "full":
        initial_assignment: dict[str, str] = {}
        initial_candidate_pool: dict[str, tuple[str, ...]] = {}
        final_candidate_pool: dict[str, tuple[str, ...]] = {}

        async def capture_initial(snapshot) -> None:
            nonlocal initial_assignment, initial_candidate_pool, final_candidate_pool
            if snapshot.assignment and not initial_assignment:
                initial_assignment = dict(snapshot.assignment)
            if snapshot.candidate_pool:
                final_candidate_pool = dict(snapshot.candidate_pool)
                if (
                    not initial_candidate_pool
                    and snapshot.events
                    and snapshot.events[-1].phase == "validate"
                ):
                    initial_candidate_pool = dict(snapshot.candidate_pool)

        snapshot = await solve_puzzle(
            run_id=f"eval-{puzzle.id}",
            puzzle=puzzle,
            provider=provider,
            lexicon=lexicon,
            clue_index=clue_index,
            settings=AgentSettings(),
            observer=capture_initial,
        )
        return ModeRunResult(
            assignment=snapshot.assignment,
            candidate_replacements=snapshot.metrics.candidate_replacements,
            initial_assignment=initial_assignment,
            elapsed_ms=snapshot.metrics.elapsed_ms,
            status=snapshot.status,
            initial_candidate_pool=initial_candidate_pool or final_candidate_pool,
            final_candidate_pool=final_candidate_pool,
        )

    raw_domains = await provider.generate_candidates(requests)
    domains = {
        entry.id: validate_domain(entry, raw_domains.get(entry.id, []))
        for entry in puzzle.entries
    }
    candidate_pool = {
        entry_id: tuple(candidate.answer for candidate in candidates)
        for entry_id, candidates in domains.items()
    }
    if mode == "single_shot":
        assignment = {
                entry.id: domains[entry.id][0].answer
                for entry in puzzle.entries
                if domains[entry.id]
        }
        return ModeRunResult(
            assignment=assignment,
            candidate_replacements=0,
            initial_assignment=assignment,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            status=None,
            initial_candidate_pool=candidate_pool,
            final_candidate_pool=candidate_pool,
        )
    if mode == "top1_constraints":
        selected: dict[str, Candidate] = {}
        for entry in puzzle.entries:
            if domains[entry.id] and candidate_fits(
                puzzle, entry.id, domains[entry.id][0].answer, selected
            ):
                selected[entry.id] = domains[entry.id][0]
        assignment = {key: value.answer for key, value in selected.items()}
        return ModeRunResult(
            assignment=assignment,
            candidate_replacements=0,
            initial_assignment=assignment,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            status=None,
            initial_candidate_pool=candidate_pool,
            final_candidate_pool=candidate_pool,
        )
    result = search_assignments(puzzle, domains)
    return ModeRunResult(
        assignment=result.words,
        candidate_replacements=0,
        initial_assignment=result.words,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        status=None,
        initial_candidate_pool=candidate_pool,
        final_candidate_pool=candidate_pool,
    )


def score_candidate_recall(
    record: PuzzleRecord,
    candidate_pool: dict[str, tuple[str, ...]],
) -> dict[str, float | bool]:
    if record.solution is None:
        raise ValueError("Gold solution is required")
    gold = {
        entry.id: "".join(record.solution[row][col] for row, col in entry.cells)
        for entry in record.puzzle.entries
    }
    ranks = []
    for entry_id, answer in gold.items():
        candidates = candidate_pool.get(entry_id, ())
        try:
            ranks.append(candidates.index(answer) + 1)
        except ValueError:
            ranks.append(None)
    total = len(ranks)

    def recall(limit: int) -> float:
        return (
            sum(rank is not None and rank <= limit for rank in ranks) / total
            if total
            else 0.0
        )

    return {
        "recall_at_1": recall(1),
        "recall_at_5": recall(5),
        "recall_at_10": recall(10),
        "mrr": (
            sum(1.0 / rank if rank is not None else 0.0 for rank in ranks) / total
            if total
            else 0.0
        ),
        "oracle_solvable": bool(ranks) and all(rank is not None for rank in ranks),
    }


def score_assignment(record: PuzzleRecord, assignment: dict[str, str]) -> dict[str, object]:
    if record.solution is None:
        raise ValueError("Gold solution is required")
    gold = {
        entry.id: "".join(record.solution[row][col] for row, col in entry.cells)
        for entry in record.puzzle.entries
    }
    correct_words = sum(assignment.get(entry_id) == answer for entry_id, answer in gold.items())
    predicted_cells: dict[tuple[int, int], set[str]] = {}
    for entry_id, answer in assignment.items():
        entry = record.puzzle.entry_map[entry_id]
        for index, cell in enumerate(entry.cells):
            if index < len(answer):
                predicted_cells.setdefault(cell, set()).add(answer[index])
    open_cells = [
        (row, col)
        for row, template_row in enumerate(record.puzzle.template)
        for col, char in enumerate(template_row)
        if char != "#"
    ]
    correct_letters = sum(
        predicted_cells.get((row, col)) == {record.solution[row][col]}
        for row, col in open_cells
    )
    consistent_intersections = sum(
        intersection.entry_a in assignment
        and intersection.entry_b in assignment
        and assignment[intersection.entry_a][intersection.index_a]
        == assignment[intersection.entry_b][intersection.index_b]
        for intersection in record.puzzle.intersections
    )
    intersection_count = len(record.puzzle.intersections)
    return {
        "letter_accuracy": correct_letters / len(open_cells) if open_cells else 0.0,
        "word_accuracy": correct_words / len(gold) if gold else 0.0,
        "full_puzzle_solved": correct_words == len(gold),
        "intersection_consistency": (
            consistent_intersections / intersection_count
            if intersection_count
            else 1.0
        ),
        "constraint_violations": count_constraint_violations(record.puzzle, assignment),
    }


def score_conflict_recovery(
    record: PuzzleRecord,
    *,
    initial_assignment: dict[str, str],
    final_assignment: dict[str, str],
) -> dict[str, object]:
    if record.solution is None:
        raise ValueError("Gold solution is required")
    gold = {
        entry.id: "".join(record.solution[row][col] for row, col in entry.cells)
        for entry in record.puzzle.entries
    }
    initially_wrong = {
        entry_id
        for entry_id, answer in initial_assignment.items()
        if gold.get(entry_id) != answer
    }
    recovered = sum(
        final_assignment.get(entry_id) == gold[entry_id]
        for entry_id in initially_wrong
    )
    opportunities = len(initially_wrong)
    return {
        "recovery_opportunities": opportunities,
        "recovered_conflicts": recovered,
        "conflict_recovery_rate": (
            recovered / opportunities if opportunities else None
        ),
    }


def summarize(results: Sequence[PuzzleEvaluation]) -> EvaluationSummary:
    if not results:
        raise ValueError("Cannot summarize an empty evaluation")
    costs = [item.estimated_cost_usd for item in results if item.estimated_cost_usd is not None]
    recovery_opportunities = sum(item.recovery_opportunities for item in results)
    recovered_conflicts = sum(item.recovered_conflicts for item in results)
    successful_solves = sum(item.full_puzzle_solved for item in results)
    complete_cost_data = len(costs) == len(results)
    return EvaluationSummary(
        mode=results[0].mode,
        puzzles=len(results),
        letter_accuracy=statistics.fmean(item.letter_accuracy for item in results),
        word_accuracy=statistics.fmean(item.word_accuracy for item in results),
        full_puzzle_solve_rate=statistics.fmean(
            1.0 if item.full_puzzle_solved else 0.0 for item in results
        ),
        intersection_consistency=statistics.fmean(
            item.intersection_consistency for item in results
        ),
        constraint_violations=sum(item.constraint_violations for item in results),
        conflict_recovery_rate=(
            recovered_conflicts / recovery_opportunities
            if recovery_opportunities
            else None
        ),
        recovery_opportunities=recovery_opportunities,
        average_model_calls=statistics.fmean(item.model_calls for item in results),
        average_input_tokens=statistics.fmean(item.input_tokens for item in results),
        average_output_tokens=statistics.fmean(item.output_tokens for item in results),
        average_total_tokens=statistics.fmean(
            item.input_tokens + item.output_tokens for item in results
        ),
        average_latency_ms=statistics.fmean(item.latency_ms for item in results),
        p50_latency_ms=statistics.median(item.latency_ms for item in results),
        average_elapsed_ms=statistics.fmean(item.elapsed_ms for item in results),
        p50_elapsed_ms=statistics.median(item.elapsed_ms for item in results),
        average_cost_usd=statistics.fmean(costs) if costs else None,
        cost_per_successful_solve_usd=(
            sum(costs) / successful_solves
            if complete_cost_data and successful_solves
            else None
        ),
        average_candidate_replacements=statistics.fmean(
            item.candidate_replacements for item in results
        ),
        initial_candidate_recall_at_1=statistics.fmean(
            item.initial_candidate_recall_at_1 for item in results
        ),
        initial_candidate_recall_at_5=statistics.fmean(
            item.initial_candidate_recall_at_5 for item in results
        ),
        initial_candidate_recall_at_10=statistics.fmean(
            item.initial_candidate_recall_at_10 for item in results
        ),
        final_candidate_recall_at_1=statistics.fmean(
            item.final_candidate_recall_at_1 for item in results
        ),
        final_candidate_recall_at_5=statistics.fmean(
            item.final_candidate_recall_at_5 for item in results
        ),
        final_candidate_recall_at_10=statistics.fmean(
            item.final_candidate_recall_at_10 for item in results
        ),
        initial_candidate_mrr=statistics.fmean(
            item.initial_candidate_mrr for item in results
        ),
        final_candidate_mrr=statistics.fmean(
            item.final_candidate_mrr for item in results
        ),
        initial_oracle_solve_rate=statistics.fmean(
            1.0 if item.initial_oracle_solvable else 0.0 for item in results
        ),
        final_oracle_solve_rate=statistics.fmean(
            1.0 if item.final_oracle_solvable else 0.0 for item in results
        ),
    )


def write_results(
    path: Path,
    results_by_mode: dict[AblationMode, list[PuzzleEvaluation]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_suffix(".json")
    raw_path.write_text(
        json.dumps(
            {
                mode: [result.model_dump(mode="json") for result in results]
                for mode, results in results_by_mode.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Crossword Agent Evaluation",
        "",
        (
            "| Mode | Puzzles | Letter accuracy | Word accuracy | Full solves | "
            "Violations | Calls | Tokens | p50 latency | Replacements | Cost |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, results in results_by_mode.items():
        summary = summarize(results)
        lines.append(
            f"| {mode} | {summary.puzzles} | {summary.letter_accuracy:.1%} | "
            f"{summary.word_accuracy:.1%} | {summary.full_puzzle_solve_rate:.1%} | "
            f"{summary.constraint_violations} | {summary.average_model_calls:.1f} | "
            f"{summary.average_input_tokens + summary.average_output_tokens:.0f} | "
            f"{summary.p50_latency_ms / 1000:.1f}s | "
            f"{summary.average_candidate_replacements:.1f} | "
            f"{_format_cost(summary.average_cost_usd)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_study_results(
    output_directory: Path,
    *,
    split: StudySplit,
    heldout_full: Sequence[PuzzleEvaluation],
    ablations: dict[AblationMode, list[PuzzleEvaluation]],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "results.md"
    write_results(report_path, ablations)
    full_summary = summarize(heldout_full)
    with report_path.open("a", encoding="utf-8") as report:
        report.write(
            "\n## Full held-out run\n\n"
            f"- Seed: `{split.seed}`\n"
            f"- Development puzzles: {len(split.development)}\n"
            f"- Separate demo puzzles: {len(split.demo)}\n"
            f"- Held-out 7×7 puzzles: {len(split.heldout_7x7)}\n"
            f"- Held-out 14×14 puzzles: {len(split.heldout_14x14)}\n"
            f"- Letter accuracy: {full_summary.letter_accuracy:.1%}\n"
            f"- Word accuracy: {full_summary.word_accuracy:.1%}\n"
            f"- Full-puzzle solve rate: {full_summary.full_puzzle_solve_rate:.1%}\n"
        )
    (output_directory / "runs.json").write_text(
        json.dumps(
            {
                "heldout_full": [
                    result.model_dump(mode="json") for result in heldout_full
                ],
                "ablations": {
                    mode: [result.model_dump(mode="json") for result in results]
                    for mode, results in ablations.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_directory / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": split.seed,
                "development": [record.puzzle.id for record in split.development],
                "demo": [record.puzzle.id for record in split.demo],
                "heldout_7x7": [record.puzzle.id for record in split.heldout_7x7],
                "heldout_14x14": [
                    record.puzzle.id for record in split.heldout_14x14
                ],
                "ablation_subset": [
                    record.puzzle.id for record in split.ablation_subset
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_bakeoff_results(path: Path, results: Sequence[ModelBakeoffResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Nebius Model Bake-off",
        "",
        "| Rank | Model | Word accuracy | Full solves | Average cost | p50 latency |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, result in enumerate(results, start=1):
        lines.append(
            f"| {rank} | {result.model} | {result.word_accuracy:.1%} | "
            f"{result.full_puzzle_solve_rate:.1%} | "
            f"{_format_cost(result.average_cost_usd)} | "
            f"{result.p50_latency_ms / 1000:.1f}s |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2),
        encoding="utf-8",
    )


def _format_cost(value: float | None) -> str:
    return f"${value:.4f}" if value is not None else "n/a"
