# Crossword Agent

A Nebius-backed crossword agent built with
[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview). The model
proposes clue answers; deterministic Python handles grid geometry, crossing
consistency, search, retries, and scoring.

## Run it in three steps

### 1. Install and configure Nebius

Requirements: Python 3.12, [`uv`](https://docs.astral.sh/uv/), and a Nebius
Token Factory API key.

```bash
uv sync --all-groups
cp .env.example .env
```

Open `.env` and set:

```bash
NEBIUS_API_KEY=your-token-here
```

The other provider settings already have working defaults.

### 2. Download and prepare CrossWordBench

Log in to Hugging Face, then create the normalized dataset consumed by the app:

```bash
uvx --from huggingface_hub hf auth login

uv run scripts/prepare_crosswordbench.py \
  --variant english_simple \
  --size 7x7 \
  --limit 20 \
  --seed 20260920 \
  --split demo \
  --output data/crosswordbench/demo-7x7.json
```

This downloads the
[CrossWordBench dataset](https://huggingface.co/datasets/HINT-lab/CrossWordBench)
from Hugging Face, converts it to the agent's JSON format, and writes 20
deterministic 7×7 puzzles.

Tell the app to use that dataset:

```bash
printf '\nCROSSWORD_DATASET_PATH=data/crosswordbench/demo-7x7.json\n' >> .env
```

The preparation script declares its own dependencies. Generated data is ignored
by Git because another person can reproduce it from the command above.

### 3. Run the local app

```bash
uv run crossword serve
```

Open <http://127.0.0.1:8000>.

The UI lets the interviewer select a puzzle, watch the agent solve it, inspect
candidate/re-query events, and review the final answers. The CLI equivalents are:

```bash
uv run crossword list-puzzles
uv run crossword solve --puzzle <PUZZLE_ID>
```

## Run the benchmark

From a second terminal, run the full agent over a seeded sample:

```bash
uv run python scripts/benchmark_agent.py \
  --dataset data/crosswordbench/demo-7x7.json \
  --count 20 \
  --concurrency 2 \
  --seed 20260920 \
  --output output/benchmark/dashboard.html
```

This writes:

- `output/benchmark/dashboard.html` — interviewer-friendly report
- `output/benchmark/dashboard.json` — raw results

The report includes exact puzzle solve rate, word/letter accuracy, crossing
consistency, candidate recall, conflict recovery, model calls, tokens, and
latency. It updates after each completed puzzle.

For a quick local smoke test, omit `CROSSWORD_DATASET_PATH`; the app falls back
to the six bundled demo puzzles. The benchmark requires a prepared dataset with
gold answers for scoring.

## Repository map

```text
src/crossword_agent/
  agent.py       LangGraph solve loop
  constraints.py deterministic grid validation and search
  providers/     Nebius structured-output provider
  api.py         local FastAPI app
  static/        browser UI
  data/          bundled smoke-test puzzles

scripts/
  prepare_crosswordbench.py  HF parquet → normalized JSON
  benchmark_agent.py         multi-puzzle benchmark
```

## Tests

```bash
uv run pytest
uv run ruff check src tests scripts
```

Cloud integration tests are opt-in because they consume model tokens:

```bash
uv run pytest -m integration
```

## Boundaries

- Text crosswords only; no OCR, images, rebuses, or cryptic clues.
- No external clue retrieval or runtime web search.
- Gold answers are used only for benchmark scoring and post-run review.
- Run state is intentionally in memory and resets when the process restarts.
