from pathlib import Path

from crossword_agent.evaluation import (
    build_study_split,
    deterministic_split,
    evaluate_records,
    score_assignment,
    score_candidate_recall,
    score_conflict_recovery,
    write_results,
    write_study_results,
)
from crossword_agent.models import Candidate
from crossword_agent.providers.fake import ScriptedCandidateProvider


def _gold_answers(record) -> dict[str, str]:
    return {
        entry.id: "".join(record.solution[row][col] for row, col in entry.cells)
        for entry in record.puzzle.entries
    }


def test_score_assignment_for_gold_and_partial(demo_record) -> None:
    gold = _gold_answers(demo_record)
    perfect = score_assignment(demo_record, gold)
    partial = score_assignment(demo_record, {"1A": gold["1A"]})

    assert perfect["letter_accuracy"] == 1.0
    assert perfect["word_accuracy"] == 1.0
    assert perfect["full_puzzle_solved"] is True
    assert perfect["intersection_consistency"] == 1.0
    assert partial["word_accuracy"] == 1 / len(gold)
    assert partial["intersection_consistency"] == 0.0


def test_conflict_recovery_counts_wrong_words_corrected_later(demo_record) -> None:
    gold = _gold_answers(demo_record)
    initial = dict(gold)
    initial["1A"] = "CAR"

    recovery = score_conflict_recovery(
        demo_record,
        initial_assignment=initial,
        final_assignment=gold,
    )

    assert recovery["recovery_opportunities"] == 1
    assert recovery["recovered_conflicts"] == 1
    assert recovery["conflict_recovery_rate"] == 1.0


def test_candidate_recall_reports_rank_and_oracle_gap(demo_record) -> None:
    gold = _gold_answers(demo_record)
    pool = {
        entry_id: (answer,)
        for entry_id, answer in gold.items()
    }
    pool["1A"] = ("CAR", gold["1A"])
    pool.pop("3D")

    recall = score_candidate_recall(demo_record, pool)

    assert recall["recall_at_1"] == (len(gold) - 2) / len(gold)
    assert recall["recall_at_5"] == (len(gold) - 1) / len(gold)
    assert recall["oracle_solvable"] is False


def test_deterministic_split_is_reproducible(demo_record) -> None:
    records = [demo_record.model_copy(deep=True) for _ in range(8)]
    for index, record in enumerate(records):
        record.puzzle = record.puzzle.model_copy(update={"id": str(index)})

    first = deterministic_split(records, dev_count=2, demo_count=2)
    second = deterministic_split(records, dev_count=2, demo_count=2)

    assert [[item.puzzle.id for item in split] for split in first] == [
        [item.puzzle.id for item in split] for split in second
    ]


def test_study_split_has_requested_sizes_and_fixed_subset(demo_record) -> None:
    records = []
    for index in range(20):
        records.append(
            demo_record.model_copy(
                update={
                    "puzzle": demo_record.puzzle.model_copy(
                        update={"id": f"dev-{index}", "width": 5, "height": 5}
                    ),
                    "split": "development",
                },
                deep=True,
            )
        )
    for size, count in ((7, 50), (14, 20)):
        for index in range(count):
            records.append(
                demo_record.model_copy(
                    update={
                        "puzzle": demo_record.puzzle.model_copy(
                            update={
                                "id": f"heldout-{size}-{index}",
                                "width": size,
                                "height": size,
                            }
                        ),
                        "split": "evaluation",
                    },
                    deep=True,
                )
            )
    demos = [
        demo_record.model_copy(
            update={
                "puzzle": demo_record.puzzle.model_copy(update={"id": f"demo-{index}"})
            },
            deep=True,
        )
        for index in range(3)
    ]

    first = build_study_split(records, demo_records=demos)
    second = build_study_split(records, demo_records=demos)

    assert len(first.development) == 20
    assert len(first.demo) == 3
    assert len(first.heldout_7x7) == 50
    assert len(first.heldout_14x14) == 20
    assert len(first.ablation_subset) == 20
    assert [item.puzzle.id for item in first.ablation_subset] == [
        item.puzzle.id for item in second.ablation_subset
    ]


async def test_all_ablation_modes_produce_results(demo_record, tmp_path: Path) -> None:
    gold = _gold_answers(demo_record)

    def provider_factory() -> ScriptedCandidateProvider:
        return ScriptedCandidateProvider(
            {
                entry_id: [[Candidate(answer=answer)]]
                for entry_id, answer in gold.items()
            }
        )

    results_by_mode = {}
    for mode in ("single_shot", "top1_constraints", "constraint_search", "full"):
        results_by_mode[mode] = await evaluate_records(
            [demo_record],
            provider_factory=provider_factory,
            mode=mode,
        )
        assert results_by_mode[mode][0].full_puzzle_solved

    report = tmp_path / "results.md"
    write_results(report, results_by_mode)
    assert "Full solves" in report.read_text(encoding="utf-8")
    assert "Replacements" in report.read_text(encoding="utf-8")
    assert report.with_suffix(".json").exists()


async def test_study_outputs_contain_no_gold_solutions(
    demo_record,
    tmp_path: Path,
) -> None:
    gold = _gold_answers(demo_record)

    def provider_factory() -> ScriptedCandidateProvider:
        return ScriptedCandidateProvider(
            {
                entry_id: [[Candidate(answer=answer)]]
                for entry_id, answer in gold.items()
            }
        )

    results = await evaluate_records(
        [demo_record],
        provider_factory=provider_factory,
        mode="full",
    )
    split = type(
        "Split",
        (),
        {
            "seed": 20260919,
            "development": (),
            "demo": (demo_record,),
            "heldout_7x7": (demo_record,),
            "heldout_14x14": (),
            "ablation_subset": (demo_record,),
        },
    )()
    write_study_results(
        tmp_path,
        split=split,
        heldout_full=results,
        ablations={"full": results},
    )

    manifest = (tmp_path / "split_manifest.json").read_text(encoding="utf-8")
    assert "solution" not in manifest
