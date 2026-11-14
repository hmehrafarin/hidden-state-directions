"""Discovering intervention runs and scoring them against the prompts."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ri.analysis.interventions import (
    ROW_SCHEMA,
    all_complete,
    baseline_disagreement,
    both_prompts_wrong,
    compare_to_baseline,
    discover_runs,
    discover_transfer_runs,
    predicted_baseline,
    summarise_run,
    transfer_cells,
)
from ri.paths import RESULTS_ROOT_ENV, stage_dir

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def rows(
    correct: dict[int, bool], dataset: str = "t", dose: float | None = 1.0, tokens: int = 5
) -> pl.DataFrame:
    idx = sorted(correct)
    return pl.DataFrame(
        {
            "dataset": [dataset] * len(idx),
            "sample_idx": idx,
            "operator": ["none" if dose is None else "rotate"] * len(idx),
            "dose": [dose] * len(idx),
            "correct": [correct[i] for i in idx],
            "n_generated_tokens": [tokens] * len(idx),
        },
        schema=ROW_SCHEMA,
    )


def baseline(correct: list[bool], dataset: str = "t") -> pl.DataFrame:
    return pl.DataFrame(
        {"dataset": [dataset] * len(correct), "sample_idx": range(len(correct)), "correct": correct}
    )


def test_a_question_without_a_row_counts_against_the_intervention() -> None:
    table = compare_to_baseline(rows({0: True, 1: True}), 1.0, baseline([True] * 4))
    row = table.row(0, named=True)
    assert row["n"] == 4
    assert row["prompt_only"] == 2
    assert row["delta"] == -0.5


def test_matching_the_prompt_everywhere_shows_no_difference() -> None:
    table = compare_to_baseline(rows({0: True, 1: False}), 1.0, baseline([True, False]))
    row = table.row(0, named=True)
    assert (row["intervention_only"], row["prompt_only"]) == (0, 0)
    assert row["p_value"] == 1.0
    assert not row["significant"]


def test_both_discordant_counts_are_reported() -> None:
    table = compare_to_baseline(
        rows({0: True, 1: True, 2: False, 3: False}), 1.0, baseline([False, False, False, True])
    )
    row = table.row(0, named=True)
    assert (row["intervention_only"], row["prompt_only"]) == (2, 1)
    assert row["delta"] == pytest.approx(0.25)


def test_only_the_requested_dose_is_compared() -> None:
    frame = pl.concat([rows({0: True, 1: True}, dose=1.0), rows({0: False, 1: False}, dose=0.5)])
    table = compare_to_baseline(frame, 1.0, baseline([False, False]))
    assert table.row(0, named=True)["intervention_only"] == 2


def test_tasks_are_tested_separately_and_corrected_together() -> None:
    base = pl.concat([baseline([False] * 20, d) for d in ("a", "b", "c")])
    frame = pl.concat(
        [
            rows(dict.fromkeys(range(20), True), "a"),
            rows(dict.fromkeys(range(20), True), "b"),
            rows(dict.fromkeys(range(20), False), "c"),
        ]
    )
    table = compare_to_baseline(frame, 1.0, base)
    assert table["dataset"].to_list() == ["a", "b", "c"]
    assert table["significant"].to_list() == [True, True, False]


def test_both_prompts_wrong_share_counts_missing_rows_as_incorrect() -> None:
    cot = baseline([False, False, True, False])
    direct = baseline([False, True, False, False])
    frame = rows({0: True, 1: False, 2: False})
    share = both_prompts_wrong(frame, 1.0, cot, direct)
    assert share == {"n_both_wrong": 2, "n_correct": 1, "share": 0.5}


def test_summary_is_per_dose_with_the_unmodified_prompt_first() -> None:
    frame = pl.concat(
        [
            rows({0: True, 1: True}, "a", dose=1.0, tokens=10),
            rows({0: False}, "b", dose=1.0, tokens=30),
            rows({0: True, 1: False}, "a", dose=None, tokens=1),
        ]
    )
    summary, per_task = summarise_run(frame, {"a": 2, "b": 4}, {"a": None, "b": None})
    assert summary["dose"].to_list() == [None, 1.0]
    treated = summary.row(1, named=True)
    assert treated["accuracy"] == pytest.approx((1.0 + 0.0) / 2)
    assert treated["coverage"] == pytest.approx(3 / 6)
    assert treated["mean_tokens"] == pytest.approx((10 + 30) / 2)
    assert per_task.columns[0] == "dose"
    task_b = per_task.filter((pl.col("dose") == 1.0) & (pl.col("dataset") == "b")).row(
        0, named=True
    )
    assert (task_b["n_scored"], task_b["n_total"], task_b["accuracy"]) == (1, 4, 0.0)


def test_baseline_prediction_weights_tasks_not_questions() -> None:
    expected = {"small": 0.5, "big": 0.5}
    assert predicted_baseline(expected, {"small": 1.0, "big": 1.0}) == pytest.approx(0.5)
    assert predicted_baseline(expected, {"small": 1.0, "big": 0.0}) == pytest.approx(0.25)


def test_a_run_reproducing_the_evaluation_is_not_flagged() -> None:
    expected = dict.fromkeys(("a", "b", "c", "d"), 0.5)
    half_covered = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
    assert baseline_disagreement(0.25, expected, half_covered) is None
    assert baseline_disagreement(0.0, expected, dict.fromkeys(expected, 0.0)) is None
    assert baseline_disagreement(0.1, expected, dict.fromkeys(expected, 1.0)) == pytest.approx(0.4)


def test_transfer_cells_score_against_the_full_target_task() -> None:
    frame = pl.concat([rows({0: True, 1: False}, "a"), rows({0: True}, "c")])
    with_direction = {"a": frozenset({0, 1, 3}), "b": None, "c": None, "e": frozenset()}
    sizes = {"s": 2, "a": 4, "b": 3, "c": 1, "e": 2}
    cells = transfer_cells(frame, sizes, with_direction, "s", [1.0])
    assert [(c["target"], c["accuracy"], c["coverage"]) for c in cells] == [
        ("a", 0.25, 0.5),
        ("b", 0.0, 0.0),
        ("c", 1.0, 1.0),
        ("e", 0.0, 0.0),
    ]
    assert [(c["n_scored"], c["n_expected"], c["complete"]) for c in cells] == [
        (2, 3, False),
        (0, None, False),
        (1, None, None),
        (0, 0, True),
    ]


def test_a_cell_is_complete_when_every_question_with_a_direction_has_a_row() -> None:
    frame = pl.concat(
        [
            rows({0: True, 2: False}, "a", dose=None),
            rows({0: True, 2: True}, "a"),
            rows({1: True}, "b"),
        ]
    )
    with_direction = {"a": frozenset({0, 2}), "b": frozenset({1, 4})}
    summary, per_task = summarise_run(frame, {"a": 3, "b": 5}, with_direction)
    cells = {
        (r["dose"], r["dataset"]): (r["n_scored"], r["n_expected"], r["complete"])
        for r in per_task.iter_rows(named=True)
    }
    assert cells == {
        (None, "a"): (2, 2, True),
        (None, "b"): (0, 2, False),
        (1.0, "a"): (2, 2, True),
        (1.0, "b"): (1, 2, False),
    }
    assert summary["complete"].to_list() == [False, False]
    _, per_task = summarise_run(frame, {"a": 3}, {"a": None})
    assert per_task["complete"].to_list() == [None, None]


def test_a_run_is_complete_only_when_every_task_is() -> None:
    assert all_complete(pl.Series([True, True])) is True
    assert all_complete(pl.Series([True, None])) is None
    assert all_complete(pl.Series([None, False])) is False


def _write(path: Path, frame: pl.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path / "part.parquet")


def test_discovery_reads_the_results_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RESULTS_ROOT_ENV, str(tmp_path))
    frame = rows({0: True})
    for name in (
        "contrastive_050_rotate_layer24",
        "task_level_050_scale_layer24",
        "recovering_mean_rotate_layer22",
    ):
        _write(stage_dir(MODEL, "intervention", name, "gsm8k"), frame)
    _write(stage_dir(MODEL, "steering", "recovering_mean_rotate_layer24", "gsm8k"), frame)
    _write(
        stage_dir(MODEL, "intervention", "task_transfer_layer24", "from_bbh_snarks", "gsm8k"), frame
    )
    stage_dir(MODEL, "intervention", "unrelated").mkdir()

    found = {(a.direction, a.operator, a.layer, a.every_token) for a in discover_runs(MODEL)}
    assert found == {
        ("contrastive_050", "rotate", 24, False),
        ("task_level_050", "scale", 24, False),
        ("recovering_mean", "rotate", 22, False),
        ("recovering_mean", "rotate", 24, True),
    }
    (transfer,) = discover_transfer_runs(MODEL)
    assert (transfer.source, transfer.direction, transfer.layer) == (
        "bbh_snarks",
        "task_level_050",
        24,
    )
    assert transfer.read().height == 1


def test_reading_a_run_concatenates_its_fragments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RESULTS_ROOT_ENV, str(tmp_path))
    leaf = stage_dir(MODEL, "intervention", "failed_mean_rotate_layer24", "bbh_snarks")
    leaf.mkdir(parents=True)
    rows({0: True}, "bbh_snarks", dose=None).write_parquet(leaf / "a.parquet")
    rows({0: False, 1: True}, "bbh_snarks").write_parquet(leaf / "b.parquet")
    (run,) = discover_runs(MODEL)
    frame = run.read()
    assert frame.height == 3
    assert frame.schema == pl.Schema(ROW_SCHEMA)


def test_a_question_written_twice_stops_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RESULTS_ROOT_ENV, str(tmp_path))
    leaf = stage_dir(MODEL, "intervention", "contrastive_050_rotate_layer24", "bbh_snarks")
    leaf.mkdir(parents=True)
    rows({0: True, 1: True}, "bbh_snarks", dose=None).write_parquet(leaf / "first.parquet")
    rows({1: False, 2: True}, "bbh_snarks", dose=None).write_parquet(leaf / "second.parquet")
    rows({1: True}, "bbh_snarks", dose=1.0).write_parquet(leaf / "third.parquet")
    (run,) = discover_runs(MODEL)
    with pytest.raises(ValueError, match="more than once") as error:
        run.read()
    message = str(error.value)
    assert str(run.path) in message
    assert "bbh_snarks/first.parquet" in message and "bbh_snarks/second.parquet" in message
    assert "third.parquet" not in message
