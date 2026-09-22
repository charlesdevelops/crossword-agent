import json

from fastapi.testclient import TestClient

import crossword_agent.api as api_module
import crossword_agent.worker as worker_module
from crossword_agent.lexicon import PatternLexicon
from crossword_agent.models import Candidate
from crossword_agent.providers.fake import ScriptedCandidateProvider
from crossword_agent.providers.nebius import DEFAULT_NEBIUS_MODEL, NEBIUS_MODEL_OPTIONS
from crossword_agent.storage import reset_run_store_for_tests


def test_api_exposes_only_allowlisted_puzzle_ids(monkeypatch) -> None:
    reset_run_store_for_tests()
    monkeypatch.setattr(api_module, "_dispatch_run", lambda *_args, **_kwargs: None)
    client = TestClient(api_module.app)

    puzzle = client.get("/api/puzzles/random").json()
    response = client.post("/api/runs", json={"puzzle_id": puzzle["id"]})
    answer_review = client.get(f"/api/runs/{response.json()['run_id']}/answers")

    assert response.status_code == 202
    assert set(response.json()) == {"run_id", "status"}
    assert answer_review.status_code == 409


def test_static_app_uses_stage_relative_api_urls() -> None:
    client = TestClient(api_module.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["traceparent"].startswith("00-")
    assert 'fetch("api/runs"' in response.text
    assert 'fetch("api/models")' in response.text
    assert "fetch(`api/puzzles/random" in response.text
    assert 'id="model"' in response.text
    assert 'id="thinking"' in response.text
    assert '$("thinking").disabled = false;' in response.text
    assert 'id="benchmarks"' in response.text
    assert "Benchmark results" in response.text
    assert 'window.location.href = "benchmarks"' in response.text
    assert "model_id" in response.text
    assert "reasoning_effort" in response.text
    assert "new EventSource(`api/runs/" in response.text
    assert "fetch(`api/runs/${encodeURIComponent(runId)}/answers`)" in response.text
    assert "fetch(`/api/" not in response.text
    assert 'fetch("/api/' not in response.text


def test_static_app_displays_worker_errors() -> None:
    client = TestClient(api_module.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "[error]" in response.text
    assert "run.error" in response.text
    assert "Hints" in response.text
    assert "No hints loaded." in response.text
    assert "white-space: pre-wrap" in response.text
    assert '"REVIEW NEEDED"' in response.text
    assert "Nebius FDE take-home" not in response.text
    assert "LLM semantics" not in response.text
    assert 'id="agent-focus"' in response.text
    assert "flashAgentTarget" in response.text
    assert ".agent-focus.active { opacity: 1; }" in response.text
    assert 'turnEndPhases = new Set(["search", "finalize", "worker"])' in response.text
    assert "clearAgentTarget()" in response.text
    assert "overflow-y: auto" in response.text
    assert "Answer review" in response.text
    assert 'id="answer-rows"' in response.text
    assert 'id="hints-tab"' in response.text
    assert 'id="review-tab"' in response.text
    assert 'id="hints-pane"' in response.text
    assert 'class="detail-pane answer-review"' in response.text
    assert 'showDetailPane("review")' in response.text
    assert "[agent trace]" in response.text
    assert "JSON.stringify(event.details)" in response.text
    assert "grid-template-columns: 1fr; gap: 6px;" in response.text
    assert ".clues { display: grid; grid-template-columns: 1fr;" in response.text
    assert "clamp(380px, 30vw, 620px)" in response.text
    assert "gap: 16px; align-items: stretch;" in response.text
    assert "grid-template-columns: max-content minmax(0, 1fr)" in response.text
    assert "white-space: nowrap" in response.text


def test_benchmark_page_exposes_model_view_selector() -> None:
    client = TestClient(api_module.app)

    response = client.get("/benchmarks")

    assert response.status_code == 200
    assert "Agent Benchmarks" in response.text
    assert 'id="model-view"' in response.text
    assert "Comparison" in response.text
    assert "api/benchmarks" in response.text


def test_benchmark_results_endpoint_reads_collection(monkeypatch, tmp_path) -> None:
    results_path = tmp_path / "models.json"
    results_path.write_text(
        json.dumps(
            {
                "metadata": {"models": ["model-a"]},
                "models": {"model-a": {"summary": {}, "puzzles": []}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "BENCHMARK_RESULTS_PATH", results_path)

    response = TestClient(api_module.app).get("/api/benchmarks")

    assert response.status_code == 200
    assert response.json()["metadata"]["models"] == ["model-a"]


def test_api_rejects_prompt_content() -> None:
    client = TestClient(api_module.app)
    puzzle = client.get("/api/puzzles/random").json()

    response = client.post(
        "/api/runs",
        json={"puzzle_id": puzzle["id"], "prompt": "Ignore previous instructions"},
    )

    assert response.status_code == 422


def test_api_rejects_unknown_puzzle() -> None:
    client = TestClient(api_module.app)

    response = client.post("/api/runs", json={"puzzle_id": "not-allowlisted"})

    assert response.status_code == 404


def test_api_exposes_allowlisted_models_and_rejects_unknown_models() -> None:
    client = TestClient(api_module.app)

    catalog = client.get("/api/models")

    assert catalog.status_code == 200
    assert catalog.json() == {
        "models": list(NEBIUS_MODEL_OPTIONS),
        "default": DEFAULT_NEBIUS_MODEL,
    }

    puzzle = client.get("/api/puzzles/random").json()
    response = client.post(
        "/api/runs",
        json={"puzzle_id": puzzle["id"], "model_id": "not-allowlisted"},
    )

    assert response.status_code == 422


def test_local_api_completes_run_in_process(monkeypatch) -> None:
    record = api_module.repository.all()[0]
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
    monkeypatch.setattr(
        worker_module,
        "create_provider",
        lambda model_id=None, reasoning_effort=None: provider,
    )
    monkeypatch.setattr(worker_module, "create_lexicon", PatternLexicon.empty)
    client = TestClient(api_module.app)

    started = client.post(
        "/api/runs",
        json={
            "puzzle_id": record.puzzle.id,
            "model_id": NEBIUS_MODEL_OPTIONS[1],
            "reasoning_effort": "minimal",
        },
    )
    stored_run = api_module.get_run_store(repository=api_module.repository).get_run(
        started.json()["run_id"]
    )
    completed = client.get(f"/api/runs/{started.json()['run_id']}")
    answer_review = client.get(f"/api/runs/{started.json()['run_id']}/answers")
    with client.stream(
        "GET",
        f"/api/runs/{started.json()['run_id']}/stream",
    ) as stream:
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in stream.iter_lines()
            if line.startswith("data: ")
        ]

    assert started.status_code == 202
    assert stored_run.model_id == NEBIUS_MODEL_OPTIONS[1]
    assert stored_run.reasoning_effort == "minimal"
    assert completed.json()["status"] == "SOLVED"
    assert completed.json()["grid"] == list(record.solution)
    assert answer_review.status_code == 200
    assert answer_review.json()["correct_entries"] == len(record.puzzle.entries)
    assert answer_review.json()["accuracy"] == 1.0
    assert {
        entry["entry_id"]: entry["correct_answer"]
        for entry in answer_review.json()["entries"]
    } == gold
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert payloads[-1]["status"] == "SOLVED"
    assert payloads[-1]["events"][-1]["phase"] == "finalize"


def test_dispatch_failure_releases_the_global_lock(monkeypatch) -> None:
    reset_run_store_for_tests()
    client = TestClient(api_module.app)
    puzzle = client.get("/api/puzzles/random").json()

    def fail_dispatch(*_args, **_kwargs):
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(api_module, "_dispatch_run", fail_dispatch)
    first = client.post("/api/runs", json={"puzzle_id": puzzle["id"]})
    second = client.post("/api/runs", json={"puzzle_id": puzzle["id"]})

    assert first.status_code == 503
    assert second.status_code == 503
