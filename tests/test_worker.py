from __future__ import annotations

import crossword_agent.worker as worker_module
from crossword_agent.lexicon import PatternLexicon
from crossword_agent.models import Candidate, SolveStatus
from crossword_agent.providers.fake import ScriptedCandidateProvider
from crossword_agent.puzzles import PuzzleRepository
from crossword_agent.storage import InMemoryRunStore


async def test_worker_claims_updates_and_releases_run(monkeypatch) -> None:
    repository = PuzzleRepository.bundled()
    record = repository.all()[0]
    gold = {
        entry.id: "".join(record.solution[row][col] for row, col in entry.cells)
        for entry in record.puzzle.entries
    }
    provider = ScriptedCandidateProvider(
        {
            entry_id: [[Candidate(answer=answer)]]
            for entry_id, answer in gold.items()
        }
    )
    store = InMemoryRunStore(max_daily_runs=2, repository=repository)
    run = store.create_run(record.puzzle.id)
    monkeypatch.setattr(
        worker_module,
        "create_provider",
        lambda model_id=None, reasoning_effort=None: provider,
    )
    monkeypatch.setattr(worker_module, "create_lexicon", PatternLexicon.empty)

    result = await worker_module.process_run(run.run_id, store=store)

    assert result is not None
    assert result.status is SolveStatus.SOLVED
    assert store.get_run(run.run_id).snapshot.grid == record.solution
    second = store.create_run(record.puzzle.id)
    assert second.status is SolveStatus.PENDING
