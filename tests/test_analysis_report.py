"""``ri task=report`` end to end, on a small synthetic results tree."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from ri.analysis import export
from ri.analysis.patch_labels import patch_label_table
from ri.analysis.reporting import model_option, resolve_model
from ri.analysis.task import ReportConfig, headline, run
from ri.intervention.results import InterventionResult, write_results
from ri.paths import RESULTS_ROOT_ENV, stage_dir

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
QWEN = "Qwen/Qwen2.5-7B-Instruct"

# Per task: direct and CoT outcomes, then the outcome of each question under the run.
OUTCOMES = {
    "gsm8k": {
        "direct": [False, False, False, True],
        "cot": [True, False, True, True],
        "treated": [True, True, None, True],
    },
    "bbh_snarks": {
        "direct": [True, False, False],
        "cot": [True, True, False],
        "treated": [True, True, True],
    },
}


class RecordingTracker:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def log_config(self, obj: object) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.update(metrics)

    def finish(self) -> None:
        pass


def _eval_task(model: str, prompt: str, task: str) -> None:
    folder = stage_dir(model, "eval", prompt, task)
    folder.mkdir(parents=True)
    n = len(OUTCOMES[task][prompt])
    pl.DataFrame(
        {
            "sample_idx": range(n),
            "correct": OUTCOMES[task][prompt],
            "n_generated_tokens": [100 if prompt == "cot" else 1] * n,
            "terminated": [True] * n,
        }
    ).write_parquet(folder / "eval.parquet")


def _eval(model: str, prompt: str) -> None:
    for task in OUTCOMES:
        _eval_task(model, prompt, task)


def _result(
    model: str, task: str, idx: int, direction: str, dose: float | None, **extra: object
) -> InterventionResult:
    correct = OUTCOMES[task]["treated"][idx] if dose is not None else OUTCOMES[task]["direct"][idx]
    return InterventionResult(
        sample_idx=idx,
        dataset=task,
        model_name=model,
        layer=24,
        direction=direction,
        operator="rotate" if dose is not None else "none",
        every_token=False,
        dose=dose,
        generated_text="x",
        n_generated_tokens=10 * (idx + 1),
        correct=bool(correct),
        n_recovering=1,
        n_failed=1,
        **extra,
    )


def _tree(model: str) -> None:
    _eval(model, "direct")
    _eval(model, "cot")
    for direction in ("contrastive_050", "task_level_050"):
        for task, outcomes in OUTCOMES.items():
            results = [
                _result(model, task, i, direction, dose)
                for i, outcome in enumerate(outcomes["treated"])
                if outcome is not None
                for dose in (None, 1.0)
            ]
            folder = stage_dir(model, "intervention", f"{direction}_rotate_layer24", task)
            write_results(results, folder)
    results = [
        _result(model, "bbh_snarks", i, "task_level_050", 1.0, source_dataset="gsm8k")
        for i in range(3)
    ]
    folder = stage_dir(model, "intervention", "task_transfer_layer24", "from_gsm8k", "bbh_snarks")
    write_results(results, folder)


def _cfg(**task: object) -> object:
    return OmegaConf.create(
        {"task": {"name": "report", **task}, "model": {"model_name": LLAMA}, "seed": 42}
    )


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(RESULTS_ROOT_ENV, str(tmp_path / "results"))
    _tree(LLAMA)
    return tmp_path / "results"


def test_report_writes_the_tables_under_the_model(tree: Path) -> None:
    tracker = RecordingTracker()
    run(_cfg(), tracker)
    out = stage_dir(LLAMA, "analysis")
    names = {p.stem for p in out.glob("*.csv")}
    assert names == {
        "baselines_per_task",
        "baselines_summary",
        "interventions_per_task",
        "interventions_summary",
        "transfer_matrix",
        "transfer_summary",
        "transfer_by_target",
        "mcnemar_per_task",
        "wilcoxon",
        "both_prompts_wrong",
    }

    summary = pl.read_csv(out / "interventions_summary.csv")
    row = summary.filter((pl.col("direction") == "contrastive_050") & (pl.col("dose") == 1.0))
    accuracy = (3 / 4 + 1) / 2
    assert row["accuracy"].item() == pytest.approx(accuracy)
    assert row["delta_direct"].item() == pytest.approx(accuracy - (1 / 4 + 1 / 3) / 2)
    assert row["delta_cot"].item() == pytest.approx(accuracy - (3 / 4 + 2 / 3) / 2)
    assert row["mean_tokens"].item() == pytest.approx((70 / 3 + 20) / 2)
    assert tracker.metrics["llama8B/contrastive_050_layer24/accuracy"] == pytest.approx(accuracy)

    wrong = pl.read_csv(out / "both_prompts_wrong.csv").filter(
        pl.col("direction") == "contrastive_050"
    )
    assert (wrong["n_both_wrong"].item(), wrong["n_correct"].item()) == (2, 2)

    matrix = pl.read_csv(out / "transfer_matrix.csv")
    assert set(zip(matrix["source"], matrix["target"], strict=True)) == {
        ("gsm8k", "bbh_snarks"),
        ("gsm8k", "gsm8k"),
        ("bbh_snarks", "bbh_snarks"),
    }
    cell = matrix.filter(pl.col("source") != pl.col("target")).row(0, named=True)
    assert cell["accuracy"] == pytest.approx(1.0)
    assert cell["delta_direct"] == pytest.approx(1 - 1 / 3)

    transfer = pl.read_csv(out / "transfer_summary.csv").row(0, named=True)
    assert (transfer["n_cells"], transfer["n_cells_below_direct"]) == (1, 0)
    assert transfer["delta_direct_mean"] == pytest.approx(1 - 1 / 3)

    tests = pl.read_csv(out / "wilcoxon.csv")
    assert set(tests["against"]) == {"cot", "direct"}
    assert tests["n_tasks"].unique().to_list() == [2]
    contrastive = tests.filter((pl.col("direction") == "contrastive_050") & (pl.col("dose") == 1.0))
    difference = dict(zip(contrastive["against"], contrastive["mean_difference"], strict=True))
    assert difference["direct"] == pytest.approx(((3 / 4 - 1 / 4) + (1 - 1 / 3)) / 2)
    assert difference["cot"] == pytest.approx(((3 / 4 - 3 / 4) + (1 - 2 / 3)) / 2)


def test_report_combines_models_into_one_directory(tree: Path, tmp_path: Path) -> None:
    _tree(QWEN)
    out = tmp_path / "combined"
    run(_cfg(models=[LLAMA, QWEN], output_dir=str(out)), RecordingTracker())
    baselines = pl.read_csv(out / "baselines_summary.csv")
    assert sorted(baselines["model"].unique()) == ["llama8B", "qwen7B"]
    assert not stage_dir(LLAMA, "analysis").exists()


def test_unknown_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="layer"):
        ReportConfig.model_validate({"name": "report", "layer": 28})


def test_a_model_is_named_by_id_or_config() -> None:
    assert resolve_model(QWEN) == QWEN
    assert resolve_model("qwen_7b") == QWEN
    assert model_option(QWEN) == "model=qwen_7b"
    assert model_option("org/other-1B") == "model.model_name=org/other-1B"


def test_a_missing_evaluation_names_the_directory_and_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RESULTS_ROOT_ENV, str(tmp_path / "results"))
    _eval(LLAMA, "direct")
    with pytest.raises(FileNotFoundError) as error:
        run(_cfg(), RecordingTracker())
    message = str(error.value)
    assert str(stage_dir(LLAMA, "eval", "cot")) in message
    assert "ri task=evaluate task.prompt=cot model=llama_8b" in message


def _sweep(model: str, task: str, labels: dict[int, list[bool]], layer: int) -> None:
    rows = [
        {"sample_idx": q, "layer": lyr, "source_position": i, "correct": label}
        for q, states in labels.items()
        for lyr in (layer - 2, layer)
        for i, label in enumerate(states)
    ]
    folder = stage_dir(model, "patch_sweep")
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(folder / f"combined_{task}.parquet")


def test_patch_labels_count_states_shares_and_coverage(tree: Path) -> None:
    _sweep(LLAMA, "gsm8k", {0: [True, False, False, False], 1: [False, False]}, 24)
    _sweep(LLAMA, "bbh_snarks", {3: [True, True], 5: [True, False], 7: [False, True, False]}, 24)
    table = patch_label_table(LLAMA, 24)
    assert table["dataset"].to_list() == ["gsm8k", "bbh_snarks", "all_tasks"]
    gsm8k, snarks, total = table.iter_rows(named=True)
    assert (gsm8k["n_questions"], gsm8k["mean_states_per_question"]) == (2, 3.0)
    assert gsm8k["recovering_share"] == pytest.approx(1 / 6)
    assert gsm8k["share_with_direction"] == pytest.approx(1 / 2)
    assert snarks["recovering_share"] == pytest.approx(4 / 7)
    assert snarks["share_with_direction"] == pytest.approx(2 / 3)
    assert total["n_questions"] == 5
    assert total["recovering_share"] == pytest.approx((1 / 6 + 4 / 7) / 2)
    assert total["share_with_direction"] == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_the_report_writes_patch_labels_at_the_primary_layer(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sweep(LLAMA, "gsm8k", {0: [True, False]}, 24)
    monkeypatch.setattr(export, "primary_intervention_layer", lambda model: 24)
    run(_cfg(), RecordingTracker())
    labels = pl.read_csv(stage_dir(LLAMA, "analysis") / "patch_labels.csv")
    assert labels["layer"].unique().to_list() == [24]
    assert labels["model"].unique().to_list() == ["llama8B"]


def test_the_headline_prints_missing_token_counts_as_na() -> None:
    summary = pl.DataFrame(
        {
            "model": ["llama8B"],
            "direction": ["contrastive_050"],
            "operator": ["rotate"],
            "layer": [28],
            "every_token": [False],
            "dose": [1.0],
            "accuracy": [0.85],
            "mean_tokens": [None],
            "delta_direct": [0.39],
            "delta_cot": [0.16],
        }
    )
    (line,) = headline({"interventions_summary": summary})
    assert "tokens n/a" in line


def _write_means(model: str, task: str, n: int, with_direction: set[int], layer: int = 24) -> None:
    folder = stage_dir(model, "directions", task)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(n)
    np.savez(
        folder / f"means_layer{layer}.npz",
        sample_idx=np.arange(n, dtype=np.int32),
        c_plus=rng.normal(size=(n, 4)).astype(np.float32),
        c_minus=rng.normal(size=(n, 4)).astype(np.float32),
        n_plus=np.ones(n, dtype=np.int32),
        n_minus=np.array([int(q in with_direction) for q in range(n)], dtype=np.int32),
    )


def _warnings(caplog: pytest.LogCaptureFixture, text: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and text in r.getMessage()
    ]


def test_incomplete_cross_task_cells_are_marked_and_left_out(
    tree: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_means(LLAMA, "gsm8k", 4, {0, 1, 3})
    _write_means(LLAMA, "bbh_snarks", 3, {0, 1, 2})
    folder = stage_dir(LLAMA, "intervention", "task_transfer_layer24", "from_bbh_snarks", "gsm8k")
    write_results(
        [_result(LLAMA, "gsm8k", 0, "task_level_050", 1.0, source_dataset="bbh_snarks")], folder
    )
    with caplog.at_level(logging.WARNING):
        run(_cfg(), RecordingTracker())
    out = stage_dir(LLAMA, "analysis")

    per_task = pl.read_csv(out / "interventions_per_task.csv")
    assert per_task["complete"].all()
    assert per_task.filter(pl.col("dataset") == "gsm8k")["n_expected"].unique().to_list() == [3]
    matrix = pl.read_csv(out / "transfer_matrix.csv")
    cells = {
        (r["source"], r["target"]): (r["n_scored"], r["n_expected"], r["complete"])
        for r in matrix.iter_rows(named=True)
    }
    assert cells[("bbh_snarks", "gsm8k")] == (1, 3, False)
    assert cells[("gsm8k", "bbh_snarks")] == (3, 3, True)
    summary = pl.read_csv(out / "transfer_summary.csv").row(0, named=True)
    assert (summary["n_tasks"], summary["n_cells"], summary["n_cells_incomplete"]) == (2, 1, 1)
    assert summary["delta_direct_mean"] == pytest.approx(1 - 1 / 3)
    assert pl.read_csv(out / "transfer_by_target.csv")["target"].to_list() == ["bbh_snarks"]
    (warning,) = _warnings(caplog, "cells lack rows")
    assert "bbh_snarks -> gsm8k (1/3)" in warning
    assert not _warnings(caplog, "cannot be checked")


def test_a_cross_task_cell_never_run_is_an_incomplete_row(
    tree: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stage_dir(LLAMA, "intervention", "task_transfer_layer24", "from_bbh_snarks").mkdir()
    with caplog.at_level(logging.WARNING):
        run(_cfg(), RecordingTracker())
    out = stage_dir(LLAMA, "analysis")

    cell = (
        pl.read_csv(out / "transfer_matrix.csv")
        .filter((pl.col("source") == "bbh_snarks") & (pl.col("target") == "gsm8k"))
        .row(0, named=True)
    )
    assert (cell["dose"], cell["n_scored"], cell["accuracy"]) == (1.0, 0, 0.0)
    assert cell["complete"] is False
    summary = pl.read_csv(out / "transfer_summary.csv").row(0, named=True)
    assert (summary["n_cells"], summary["n_cells_incomplete"], summary["n_sources"]) == (1, 1, 1)
    assert pl.read_csv(out / "transfer_by_target.csv")["target"].to_list() == ["bbh_snarks"]
    (warning,) = _warnings(caplog, "cells lack rows")
    assert "bbh_snarks -> gsm8k (0/?)" in warning


def test_without_means_completeness_is_unknown_and_said_once(
    tree: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        run(_cfg(), RecordingTracker())
    out = stage_dir(LLAMA, "analysis")
    assert pl.read_csv(out / "interventions_per_task.csv")["n_expected"].is_null().all()
    summary = pl.read_csv(out / "transfer_summary.csv").row(0, named=True)
    assert (summary["n_cells"], summary["n_cells_unchecked"]) == (1, 1)
    (warning,) = _warnings(caplog, "cannot be checked")
    assert "layer 24: every task" in warning


def test_a_task_missing_under_one_prompt_leaves_every_average(
    tree: Path, caplog: pytest.LogCaptureFixture
) -> None:
    shutil.rmtree(stage_dir(LLAMA, "eval", "cot", "bbh_snarks"))
    folder = stage_dir(LLAMA, "intervention", "task_transfer_layer24", "from_bbh_snarks", "gsm8k")
    write_results(
        [_result(LLAMA, "gsm8k", 0, "task_level_050", 1.0, source_dataset="bbh_snarks")], folder
    )
    with caplog.at_level(logging.WARNING):
        run(_cfg(), RecordingTracker())
    out = stage_dir(LLAMA, "analysis")
    (warning,) = _warnings(caplog, "evaluated under both prompts")
    assert "bbh_snarks" in warning

    for table in ("baselines_summary", "interventions_summary", "wilcoxon", "both_prompts_wrong"):
        assert pl.read_csv(out / f"{table}.csv")["n_tasks"].unique().to_list() == [1], table
    assert not (out / "transfer_summary.csv").exists()
    row = (
        pl.read_csv(out / "interventions_summary.csv")
        .filter((pl.col("direction") == "contrastive_050") & (pl.col("dose") == 1.0))
        .row(0, named=True)
    )
    assert row["accuracy"] == pytest.approx(3 / 4)
    assert (row["n_rows"], row["coverage"]) == (3, pytest.approx(3 / 4))
    assert pl.read_csv(out / "interventions_summary.csv")["coverage"].max() <= 1
    assert row["delta_cot"] == pytest.approx(0.0)
    assert row["delta_direct"] == pytest.approx(3 / 4 - 1 / 4)
    assert set(pl.read_csv(out / "baselines_per_task.csv")["dataset"]) == {"gsm8k"}
    assert not (out / "transfer_matrix.csv").exists()


def test_cross_task_runs_cover_only_the_tasks_under_both_prompts(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        OUTCOMES,
        "bbh_navigate",
        {"direct": [False, True], "cot": [True, True], "treated": [True, True]},
    )
    for prompt in ("direct", "cot"):
        _eval_task(LLAMA, prompt, "bbh_navigate")
    shutil.rmtree(stage_dir(LLAMA, "eval", "cot", "bbh_snarks"))
    transfer = stage_dir(LLAMA, "intervention", "task_transfer_layer24")
    for source, target in [("gsm8k", "bbh_navigate"), ("bbh_snarks", "gsm8k")]:
        write_results(
            [
                _result(LLAMA, target, i, "task_level_050", 1.0, source_dataset=source)
                for i in range(2)
            ],
            transfer / f"from_{source}" / target,
        )
    run(_cfg(), RecordingTracker())
    out = stage_dir(LLAMA, "analysis")

    matrix = pl.read_csv(out / "transfer_matrix.csv")
    assert set(matrix["source"]) | set(matrix["target"]) == {"gsm8k", "bbh_navigate"}
    summary = pl.read_csv(out / "transfer_summary.csv").row(0, named=True)
    assert (summary["n_tasks"], summary["n_sources"], summary["n_cells"]) == (2, 1, 1)
