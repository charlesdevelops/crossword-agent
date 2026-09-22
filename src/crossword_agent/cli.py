from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from crossword_agent.agent import solve_puzzle
from crossword_agent.config import load_local_env
from crossword_agent.evaluation import (
    build_study_split,
    evaluate_records,
    evaluate_study,
    load_normalized_dataset,
    run_model_bakeoff,
    write_bakeoff_results,
    write_results,
    write_study_results,
)
from crossword_agent.logging_config import configure_logging
from crossword_agent.providers.base import CandidateProvider
from crossword_agent.providers.nebius import NEBIUS_MODEL_OPTIONS, NebiusCandidateProvider
from crossword_agent.puzzles import PuzzleRepository
from crossword_agent.runtime import create_lexicon
from crossword_agent.tracing import configure_tracing

load_local_env()
configure_logging()
configure_tracing()
logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, help="Constraint-solving crossword agent")


def _provider() -> CandidateProvider:
    return NebiusCandidateProvider()


def _repository() -> PuzzleRepository:
    try:
        return PuzzleRepository.configured()
    except (OSError, KeyError, TypeError, ValueError) as error:
        logger.exception("Unable to load configured crossword dataset")
        typer.echo(f"Dataset error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command()
def solve(
    puzzle: Annotated[str, typer.Option("--puzzle", help="Configured puzzle ID")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Solve one bundled puzzle."""

    try:
        definition = _repository().get(puzzle).puzzle
    except KeyError as error:
        raise typer.BadParameter(str(error)) from error
    result = asyncio.run(
        solve_puzzle(
            run_id="cli",
            puzzle=definition,
            provider=_provider(),
            lexicon=create_lexicon(),
        )
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"{definition.title}: {result.status.value}")
    typer.echo("\n".join(result.grid))
    typer.echo(
        f"{len(result.assignment)}/{len(definition.entries)} entries, "
        f"{result.metrics.model_calls} model calls, "
        f"{result.metrics.elapsed_ms / 1000:.1f}s"
    )
    if result.error:
        typer.echo(f"Error: {result.error}", err=True)


@app.command("list-puzzles")
def list_puzzles() -> None:
    """List configured puzzle IDs."""

    for record in _repository().all():
        typer.echo(f"{record.puzzle.id}  {record.puzzle.title}")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Run the local web app."""

    import uvicorn

    repository = _repository()
    configured_dataset = os.getenv("CROSSWORD_DATASET_PATH")
    source = (
        str(Path(configured_dataset).resolve())
        if configured_dataset
        else "the bundled demo dataset"
    )
    typer.echo(f"Loaded {len(repository.all())} puzzles from {source}")
    uvicorn.run("crossword_agent.api:app", host=host, port=port, reload=False)


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("output/evaluation/results.md"),
    modes: Annotated[
        str,
        typer.Option(help="Comma-separated ablations"),
    ] = "single_shot,top1_constraints,constraint_search,full",
) -> None:
    """Run reproducible ablations against a normalized dataset."""

    records = load_normalized_dataset(dataset)
    selected_modes = [mode.strip() for mode in modes.split(",") if mode.strip()]
    allowed = {"single_shot", "top1_constraints", "constraint_search", "full"}
    if not set(selected_modes) <= allowed:
        raise typer.BadParameter(f"Modes must be drawn from {sorted(allowed)}")
    results_by_mode = {}
    for mode in selected_modes:
        typer.echo(f"Running {mode} on {len(records)} puzzles")
        results_by_mode[mode] = asyncio.run(
            evaluate_records(
                records,
                provider_factory=_provider,
                mode=mode,  # type: ignore[arg-type]
                lexicon=create_lexicon(),
            )
        )
    write_results(output, results_by_mode)  # type: ignore[arg-type]
    typer.echo(json.dumps({"report": str(output), "raw": str(output.with_suffix('.json'))}))


@app.command("evaluate-study")
def evaluate_study_command(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("output/evaluation"),
    seed: Annotated[int, typer.Option()] = 20260919,
) -> None:
    """Run the specified 20/3/50/20 held-out study and fixed ablations."""

    split = build_study_split(
        load_normalized_dataset(dataset),
        demo_records=_repository().all(),
        seed=seed,
    )
    typer.echo(
        f"Running full agent on {len(split.heldout)} held-out puzzles and "
        f"ablations on {len(split.ablation_subset)}"
    )
    heldout_full, ablations = asyncio.run(
        evaluate_study(
            split,
            provider_factory=_provider,
            lexicon=create_lexicon(),
        )
    )
    write_study_results(
        output,
        split=split,
        heldout_full=heldout_full,
        ablations=ablations,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output / "results.md"),
                "runs": str(output / "runs.json"),
                "split": str(output / "split_manifest.json"),
            }
        )
    )


@app.command()
def bakeoff(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("output/evaluation/bakeoff.md"),
    models: Annotated[
        str,
        typer.Option(help="Comma-separated Nebius model IDs"),
    ] = ",".join(NEBIUS_MODEL_OPTIONS),
    seed: Annotated[int, typer.Option()] = 20260919,
) -> None:
    """Rank Nebius models on the 20-puzzle development split."""

    split = build_study_split(
        load_normalized_dataset(dataset),
        demo_records=_repository().all(),
        seed=seed,
    )
    selected_models = [model.strip() for model in models.split(",") if model.strip()]
    if not selected_models:
        raise typer.BadParameter("At least one model ID is required")
    results = asyncio.run(
        run_model_bakeoff(
            split.development,
            models=selected_models,
            provider_factory=lambda model: NebiusCandidateProvider(model_id=model),
            lexicon=create_lexicon(),
        )
    )
    write_bakeoff_results(output, results)
    typer.echo(
        json.dumps(
            {
                "winner": results[0].model,
                "report": str(output),
                "raw": str(output.with_suffix(".json")),
            }
        )
    )
