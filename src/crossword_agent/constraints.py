from __future__ import annotations

from dataclasses import dataclass

from crossword_agent.models import Candidate, Entry, PuzzleDefinition


@dataclass(frozen=True)
class SearchHypothesis:
    assignment: dict[str, Candidate]
    score: float
    complete: bool

    @property
    def words(self) -> dict[str, str]:
        return {entry_id: candidate.answer for entry_id, candidate in self.assignment.items()}


@dataclass(frozen=True)
class SearchResult:
    assignment: dict[str, Candidate]
    nodes_visited: int
    complete: bool
    hypotheses: tuple[SearchHypothesis, ...] = ()

    @property
    def words(self) -> dict[str, str]:
        return {entry_id: candidate.answer for entry_id, candidate in self.assignment.items()}

    @property
    def score_margin(self) -> float | None:
        if len(self.hypotheses) < 2:
            return None
        first, second = self.hypotheses[:2]
        if len(first.assignment) != len(second.assignment):
            return None
        return first.score - second.score


def normalize_answer(answer: str) -> str:
    return "".join(char for char in answer.upper() if char.isalpha())


def validate_domain(
    entry: Entry,
    candidates: list[Candidate],
    *,
    pattern: str | None = None,
) -> list[Candidate]:
    normalized_pattern = pattern or "?" * entry.length
    accepted: dict[str, Candidate] = {}
    for candidate in candidates:
        answer = normalize_answer(candidate.answer)
        if len(answer) != entry.length:
            continue
        if any(
            known != "?" and known != actual
            for known, actual in zip(normalized_pattern, answer, strict=True)
        ):
            continue
        updated = candidate.model_copy(update={"answer": answer})
        existing = accepted.get(answer)
        accepted[answer] = updated if existing is None else _merge_candidate(existing, updated)
    return sorted(accepted.values(), key=lambda item: (-item.score, item.answer))


def candidate_fits(
    puzzle: PuzzleDefinition,
    entry_id: str,
    answer: str,
    assignment: dict[str, Candidate],
) -> bool:
    entry = puzzle.entry_map[entry_id]
    cell_letters = {
        cell: assigned.answer[index]
        for assigned_id, assigned in assignment.items()
        for index, cell in enumerate(puzzle.entry_map[assigned_id].cells)
    }
    return all(
        cell not in cell_letters or cell_letters[cell] == answer[index]
        for index, cell in enumerate(entry.cells)
    )


def entry_pattern(
    puzzle: PuzzleDefinition,
    entry_id: str,
    assignment: dict[str, Candidate] | dict[str, str],
) -> str:
    entry = puzzle.entry_map[entry_id]
    cell_letters: dict[tuple[int, int], str] = {}
    for assigned_id, assigned in assignment.items():
        if assigned_id == entry_id:
            continue
        answer = assigned.answer if isinstance(assigned, Candidate) else assigned
        for index, cell in enumerate(puzzle.entry_map[assigned_id].cells):
            cell_letters[cell] = answer[index]
    return "".join(cell_letters.get(cell, "?") for cell in entry.cells)


def render_grid(
    puzzle: PuzzleDefinition,
    assignment: dict[str, Candidate] | dict[str, str],
    *,
    tolerate_conflicts: bool = False,
) -> tuple[str, ...]:
    rows = [list(row) for row in puzzle.template]
    conflict_cells: set[tuple[int, int]] = set()
    for entry_id, assigned in assignment.items():
        answer = assigned.answer if isinstance(assigned, Candidate) else assigned
        entry = puzzle.entry_map[entry_id]
        for index, (row, col) in enumerate(entry.cells):
            if (row, col) in conflict_cells:
                continue
            existing = rows[row][col]
            if existing not in {".", answer[index]}:
                if tolerate_conflicts:
                    conflict_cells.add((row, col))
                    rows[row][col] = "?"
                    continue
                raise ValueError(f"Inconsistent assignment at {(row, col)}")
            rows[row][col] = answer[index]
    return tuple("".join(row) for row in rows)


def count_constraint_violations(
    puzzle: PuzzleDefinition,
    assignment: dict[str, Candidate] | dict[str, str],
) -> int:
    violations = 0
    words = {
        entry_id: candidate.answer if isinstance(candidate, Candidate) else candidate
        for entry_id, candidate in assignment.items()
    }
    for entry_id, answer in words.items():
        if len(answer) != puzzle.entry_map[entry_id].length:
            violations += 1
    for intersection in puzzle.intersections:
        if intersection.entry_a not in words or intersection.entry_b not in words:
            continue
        if (
            words[intersection.entry_a][intersection.index_a]
            != words[intersection.entry_b][intersection.index_b]
        ):
            violations += 1
    return violations


def search_assignments(
    puzzle: PuzzleDefinition,
    domains: dict[str, list[Candidate]],
    *,
    max_nodes: int = 25_000,
    n_best: int = 5,
) -> SearchResult:
    entry_ids = tuple(entry.id for entry in puzzle.entries)
    crossings_by_entry: dict[str, tuple[tuple[str, int, int], ...]] = {
        entry_id: tuple(
            (
                intersection.entry_b
                if intersection.entry_a == entry_id
                else intersection.entry_a,
                intersection.index_a
                if intersection.entry_a == entry_id
                else intersection.index_b,
                intersection.index_b
                if intersection.entry_a == entry_id
                else intersection.index_a,
            )
            for intersection in puzzle.intersections
            if entry_id in {intersection.entry_a, intersection.entry_b}
        )
        for entry_id in entry_ids
    }
    crossing_degree = {
        entry_id: sum(
            entry_id in {intersection.entry_a, intersection.entry_b}
            for intersection in puzzle.intersections
        )
        for entry_id in entry_ids
    }
    hypotheses: dict[tuple[tuple[str, str], ...], SearchHypothesis] = {}
    nodes = 0

    def assignment_score(current: dict[str, Candidate]) -> float:
        return sum(candidate.score for candidate in current.values())

    def hypothesis_key(current: dict[str, Candidate]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((key, value.answer) for key, value in current.items()))

    def rank_key(hypothesis: SearchHypothesis) -> tuple[object, ...]:
        return (
            -len(hypothesis.assignment),
            -hypothesis.score,
            hypothesis_key(hypothesis.assignment),
        )

    def remember(current: dict[str, Candidate]) -> None:
        key = hypothesis_key(current)
        candidate = SearchHypothesis(
            assignment=dict(current),
            score=assignment_score(current),
            complete=len(current) == len(entry_ids),
        )
        existing = hypotheses.get(key)
        if existing is None or candidate.score > existing.score:
            hypotheses[key] = candidate
        if len(hypotheses) > max(n_best * 4, 20):
            keep = sorted(hypotheses.values(), key=rank_key)[: max(n_best * 2, 10)]
            hypotheses.clear()
            hypotheses.update((hypothesis_key(item.assignment), item) for item in keep)

    def recurse(
        current: dict[str, Candidate],
        remaining: tuple[str, ...],
    ) -> None:
        nonlocal nodes
        if nodes >= max_nodes:
            return
        nodes += 1
        remember(current)
        if not remaining or len(current) == len(entry_ids):
            return
        ranked = sorted(hypotheses.values(), key=rank_key)
        minimum_coverage = (
            len(ranked[min(n_best, len(ranked)) - 1].assignment)
            if ranked
            else 0
        )
        if len(current) + len(remaining) < minimum_coverage:
            return

        options_by_entry = {
            entry_id: [
                candidate
                for candidate in domains.get(entry_id, [])
                if all(
                    other_id not in current
                    or current[other_id].answer[other_position]
                    == candidate.answer[current_position]
                    for other_id, current_position, other_position in crossings_by_entry[
                        entry_id
                    ]
                )
            ]
            for entry_id in remaining
        }
        selected = min(
            remaining,
            key=lambda entry_id: (
                len(options_by_entry[entry_id]),
                -crossing_degree[entry_id],
                entry_id,
            ),
        )
        next_remaining = tuple(entry_id for entry_id in remaining if entry_id != selected)
        for candidate in options_by_entry[selected]:
            current[selected] = candidate
            recurse(current, next_remaining)
            current.pop(selected)
            if nodes >= max_nodes:
                return
        recurse(current, next_remaining)

    recurse({}, entry_ids)
    ranked_hypotheses = tuple(sorted(hypotheses.values(), key=rank_key)[:n_best])
    best = ranked_hypotheses[0] if ranked_hypotheses else SearchHypothesis({}, 0.0, False)
    return SearchResult(
        assignment=best.assignment,
        nodes_visited=nodes,
        complete=best.complete,
        hypotheses=ranked_hypotheses,
    )


def _merge_candidate(first: Candidate, second: Candidate) -> Candidate:
    sources = sorted(set(first.source.split("+")) | set(second.source.split("+")))
    return Candidate(
        answer=first.answer,
        rank=min(first.rank, second.rank),
        lexical_score=max(first.lexical_score, second.lexical_score),
        source="+".join(sources),
    )
