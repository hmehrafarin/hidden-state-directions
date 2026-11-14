"""Loading, cross-fitting, transfer, report tables and the runner, on synthetic caches."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from ri.activations.store import (
    activation_sample_path,
    read_activation_hidden_states,
    write_activation_sample,
)
from ri.paths import RESULTS_ROOT_ENV
from ri.probe import artifacts, report
from ri.probe import task as probe_task
from ri.probe.artifacts import assign_folds, probe_dir, transfer_path
from ri.probe.data import TaskLayerData, balanced_pool, load_task_layer
from ri.probe.train import _CrossFitter, cross_fit, train_probe, write_artifacts
from ri.probe.transfer import evaluate_transfer, write_transfer
from ri.tasks import TASKS as ALL_TASKS

CPU = torch.device("cpu")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BAND = [26, 28]
TASKS = ("gsm8k", "bbh_boolean_expressions", "bbh_web_of_lies")
SEED = 42
CONF = Path(__file__).resolve().parents[1] / "ri" / "conf" / "task"


def _write_cache(root: Path) -> None:
    """Per task, 30-38 questions of 4-15 CoT states with a linear signal and a position trend."""
    for t, task in enumerate(TASKS):
        rng = np.random.default_rng(t)
        direction = {layer: rng.standard_normal(12) for layer in BAND}
        for q in rng.choice(400, 30 + 4 * t, replace=False):
            n = int(rng.integers(4, 16))
            states, labels = {}, {}
            for layer in BAND:
                X = rng.standard_normal((n, 12)) * 2.0
                logit = X @ direction[layer] * 0.4 - 0.15 * np.arange(n) + rng.normal(0, 1.5, n)
                states[layer], labels[layer] = X, logit > 0
            write_activation_sample(
                activation_sample_path(root, MODEL, task, int(q)),
                sample_idx=int(q),
                layer_hidden_states=states,
                layer_labels=labels,
                metadata={"model_name": MODEL, "dataset": task, "num_tokens": n},
            )


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> str:
    root = tmp_path_factory.mktemp("hs_cache")
    _write_cache(root)
    return str(root)


@pytest.fixture(scope="module")
def trained(cache: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Every synthetic task's probe at both layers, under a module-wide results root."""
    root = tmp_path_factory.mktemp("results")
    mp = pytest.MonkeyPatch()
    mp.setenv(RESULTS_ROOT_ENV, str(root))
    for layer in BAND:
        for task in TASKS:
            train_probe(MODEL, task, layer, hs_cache_dir=cache, seed=SEED, device=CPU)
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def transfers(trained: Path, cache: str) -> dict[int, pl.DataFrame]:
    """The transfer matrix of the synthetic tasks at both layers, written next to the probes."""
    matrices = {}
    for layer in BAND:
        matrices[layer] = evaluate_transfer(
            MODEL, layer, hs_cache_dir=cache, seed=SEED, device=CPU, tasks=list(TASKS)
        )
        write_transfer(MODEL, layer, matrices[layer])
    return matrices


def test_rows_are_ordered_folded_and_read_from_the_cache(cache: str) -> None:
    data = load_task_layer(MODEL, cache, "gsm8k", 28, seed=SEED)
    rows = data.rows
    assert rows.equals(rows.sort("sample_idx", "source_position"))
    folds = assign_folds("gsm8k", rows["sample_idx"].unique().to_list())
    assert rows["fold"].to_list() == [folds[q] for q in rows["sample_idx"].to_list()]
    assert data.X.dtype == np.float32 and len(data.X) == rows.height
    first = rows["sample_idx"][0]
    n_first = int((rows["sample_idx"] == first).sum())
    stored = read_activation_hidden_states(activation_sample_path(cache, MODEL, "gsm8k", first), 28)
    np.testing.assert_array_equal(data.X[:n_first], stored)


def test_a_layer_missing_from_the_cache_is_refused(cache: str) -> None:
    with pytest.raises(ValueError, match="not in the activation cache"):
        load_task_layer(MODEL, cache, "gsm8k", 30, seed=SEED)


def test_origin_labels_are_fixed_per_position_at_the_task_rate(cache: str) -> None:
    rows = load_task_layer(MODEL, cache, "bbh_boolean_expressions", 28, seed=SEED).rows
    per_position = rows.group_by("source_position").agg(pl.col("origin_label").n_unique())
    assert per_position["origin_label"].max() == 1
    largest_position = rows.group_by("source_position").len()["len"].max()
    assert abs(int(rows["origin_label"].sum()) - int(rows["label"].sum())) <= largest_position


def test_within_position_labels_keep_each_position_and_folds_rate(cache: str) -> None:
    rows = load_task_layer(MODEL, cache, "bbh_boolean_expressions", 28, seed=SEED).rows
    rates = rows.group_by("source_position", "fold").agg(
        pl.col("label").sum().alias("real"), pl.col("within_position_label").sum().alias("control")
    )
    assert rates["real"].to_list() == rates["control"].to_list()
    assert not rows["within_position_label"].equals(rows["label"])


def test_each_fold_of_the_training_pool_is_class_balanced(cache: str) -> None:
    rows = load_task_layer(MODEL, cache, "bbh_web_of_lies", 26, seed=SEED).rows
    pool = rows.filter(pl.col("in_pool"))
    assert pool.group_by("fold").agg(pl.col("label").mean())["label"].to_list() == pytest.approx(
        [0.5] * 5
    )


def _task(labels_by_question: bool, seed: int = 0) -> tuple[TaskLayerData, np.ndarray]:
    """60 questions x 12 positions. Labels follow a fixed direction (signal), or each
    question has its own random label and a strong identity vector (memorisable)."""
    rng = np.random.default_rng(seed)
    n_q, per_q, h = 60, 12, 96
    sample_idx = np.repeat(np.arange(n_q), per_q)
    identity = rng.standard_normal((n_q, h)) * 3.0
    X = identity[sample_idx] + rng.standard_normal((n_q * per_q, h))
    if labels_by_question:
        label = (rng.random(n_q) < 0.5)[sample_idx]
    else:
        label = X @ rng.standard_normal(h) + 2.0 * rng.standard_normal(len(X)) > 0
    folds = assign_folds("synthetic", range(n_q))
    fold = np.array([folds[q] for q in sample_idx])
    rows = pl.DataFrame(
        {
            "sample_idx": sample_idx.astype(np.int32),
            "source_position": np.tile(np.arange(per_q), n_q).astype(np.int32),
            "fold": fold.astype(np.int8),
            "label": label,
            "origin_label": rng.random(len(X)) < 0.5,
            "within_position_label": rng.permutation(label),
            "in_pool": balanced_pool(label, fold, 5, seed),
        }
    )
    return TaskLayerData("synthetic", 0, rows, X.astype(np.float32)), label


@pytest.fixture(scope="module")
def signal_fit() -> tuple[TaskLayerData, dict]:
    """A task whose labels follow one direction, cross-fitted once for the tests below."""
    data, _ = _task(labels_by_question=False)
    return data, cross_fit(data, device=CPU, l2_grid=(1.0, 1e-1, 1e-2))


def test_out_of_fold_scores_cannot_memorise_questions() -> None:
    """A probe that saw a question can memorise its identity; out-of-fold ones cannot."""
    data, label = _task(labels_by_question=True)
    result = cross_fit(data, device=CPU, l2_grid=(1.0, 1e-2, 1e-4))
    assert abs(result["summary"]["out_of_fold"]["auroc"] - 0.5) < 0.15
    full = result["full"]["real"]
    in_sample = data.X.astype(np.float64) @ full.weight + full.bias
    assert roc_auc_score(label, in_sample) > 0.9


def test_a_real_direction_is_recovered_out_of_fold(signal_fit: tuple[TaskLayerData, dict]) -> None:
    _, result = signal_fit
    summary = result["summary"]
    assert summary["out_of_fold"]["auroc"] > 0.8
    assert all(f["l2_penalty"] in (1.0, 1e-1, 1e-2) for f in summary["folds"])
    assert not np.isnan(result["oof"]["real"]).any()


def test_the_penalty_grid_must_descend() -> None:
    data, _ = _task(labels_by_question=False)
    with pytest.raises(ValueError, match="descending"):
        cross_fit(data, device=CPU, l2_grid=(1e-2, 1.0))


def test_raw_space_scores_equal_standardised_scores() -> None:
    data, _ = _task(labels_by_question=False)
    fitter = _CrossFitter(data, CPU)
    ts = fitter.training_set([0, 1, 2, 3])
    (probe,) = fitter.fit_path(ts, "real", [0.1])
    rows = np.flatnonzero(fitter.fold == 4)
    raw = fitter.score(rows, [probe])[:, 0]
    weight_z = probe.weight * ts.sd.numpy()
    bias_z = probe.bias + float((ts.mean.numpy() * probe.weight).sum())
    z = (data.X[rows].astype(np.float64) - ts.mean.numpy()) / ts.sd.numpy()
    np.testing.assert_allclose(raw, z @ weight_z + bias_z, rtol=1e-9, atol=1e-9)


def test_the_pattern_is_the_covariance_times_the_filter() -> None:
    data, _ = _task(labels_by_question=False)
    fitter = _CrossFitter(data, CPU)
    ts = fitter.training_set([0, 1, 2, 3])
    (probe,) = fitter.fit_path(ts, "real", [0.1])
    encoding = fitter.encoding(ts, probe)
    X = data.X[ts.pool_rows].astype(np.float64)
    y = fitter.labels["real"][ts.pool_rows].astype(bool)
    np.testing.assert_allclose(
        encoding["pattern"], np.cov(X.T, bias=True) @ probe.weight, rtol=1e-8
    )
    np.testing.assert_allclose(
        encoding["mean_difference"], X[y].mean(axis=0) - X[~y].mean(axis=0), rtol=1e-8, atol=1e-10
    )


def test_artifacts_hold_the_probes_and_are_never_overwritten(
    tmp_path: Path, signal_fit: tuple[TaskLayerData, dict]
) -> None:
    data, result = signal_fit
    metrics = write_artifacts(tmp_path, "test-model", data, result, meta={})
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    with np.load(tmp_path / "directions.npz") as npz:
        np.testing.assert_array_equal(npz["weight_raw"], result["full"]["real"].weight)
        assert npz["fold_within_position_weight_raw"].shape == (5, data.X.shape[1])
    predictions = pl.read_parquet(tmp_path / "out_of_fold_predictions.parquet")
    np.testing.assert_allclose(
        predictions["score"].to_numpy(), result["oof"]["real"].astype(np.float32)
    )
    assert pl.read_parquet(tmp_path / "question_folds.parquet").height == 60
    with pytest.raises(FileExistsError):
        write_artifacts(tmp_path, "test-model", data, result, meta={})


def test_saved_probes_reproduce_the_out_of_fold_auroc(trained: Path) -> None:
    metrics = json.loads((probe_dir(MODEL, 28, "gsm8k") / "metrics.json").read_text())
    predictions = pl.read_parquet(probe_dir(MODEL, 28, "gsm8k") / "out_of_fold_predictions.parquet")
    auroc = roc_auc_score(predictions["label"], predictions["score"])
    assert metrics["out_of_fold"]["auroc"] == pytest.approx(auroc, abs=1e-6)
    assert metrics["out_of_fold"]["selectivity_within_position"] == pytest.approx(
        metrics["out_of_fold"]["auroc"] - metrics["out_of_fold"]["control_within_position_auroc"]
    )
    assert metrics["full_refit"]["l2_penalty"] in metrics["l2_grid"]


def test_the_transfer_matrix_scores_full_refits_off_the_diagonal(
    transfers: dict[int, pl.DataFrame], cache: str
) -> None:
    matrix = transfers[28]
    assert matrix.height == len(TASKS) ** 2
    diagonal = matrix.filter(pl.col("is_diagonal"))
    for row in diagonal.iter_rows(named=True):
        metrics = json.loads(
            (probe_dir(MODEL, 28, row["source_task"]) / "metrics.json").read_text()
        )
        assert row["auroc"] == pytest.approx(metrics["out_of_fold"]["auroc"], abs=1e-6)
        assert row["origin_control_auroc"] is not None
    assert matrix.filter(~pl.col("is_diagonal"))["origin_control_auroc"].null_count() == 6

    target = load_task_layer(MODEL, cache, "bbh_boolean_expressions", 28, seed=SEED)
    X = target.X.astype(np.float64)
    with np.load(probe_dir(MODEL, 28, "gsm8k") / "directions.npz") as npz:
        scores = X @ npz["weight_raw"] + float(npz["bias_raw"])
        control = X @ npz["within_position_weight_raw"] + float(npz["within_position_bias_raw"])
    cell = matrix.filter(
        (pl.col("source_task") == "gsm8k") & (pl.col("target_task") == "bbh_boolean_expressions")
    ).row(0, named=True)
    auroc = roc_auc_score(target.rows["label"], scores)
    control_auroc = roc_auc_score(target.rows["within_position_label"], control)
    assert cell["auroc"] == pytest.approx(auroc)
    assert cell["within_position_control_auroc"] == pytest.approx(control_auroc)
    assert cell["selectivity_within_position"] == pytest.approx(auroc - control_auroc)


def test_a_transfer_matrix_is_written_once(transfers: dict[int, pl.DataFrame]) -> None:
    path = transfer_path(MODEL, 28)
    assert pl.read_parquet(path).equals(transfers[28])
    summary = json.loads(path.with_name("transfer_summary.json").read_text())
    assert summary["off_diagonal_cells"] == 6
    with pytest.raises(FileExistsError):
        write_transfer(MODEL, 28, transfers[28])


def test_a_missing_probe_blocks_the_transfer_matrix(trained: Path, cache: str) -> None:
    with pytest.raises(RuntimeError, match="probes missing"):
        evaluate_transfer(
            MODEL, 28, hs_cache_dir=cache, seed=SEED, device=CPU, tasks=["strategyqa"]
        )


@pytest.fixture
def report_inputs(
    trained: Path, transfers: dict[int, pl.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Probes, transfer matrices and evaluations of the synthetic tasks, as the report sees them."""
    for prompt in ("cot", "direct"):
        for i, task in enumerate(TASKS):
            leaf = trained / "llama8B" / "eval" / prompt / task
            leaf.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "correct": [j < (i + 1 if prompt == "cot" else 2 * i) for j in range(4)],
                    "n_generated_tokens": [10] * 4,
                    "terminated": [True] * 4,
                }
            ).write_parquet(leaf / "eval.parquet")
    monkeypatch.setattr(report, "TASKS", TASKS)
    monkeypatch.setattr(artifacts, "TASKS", TASKS)
    monkeypatch.setattr(report, "late_band", lambda model: BAND)
    monkeypatch.setattr(report, "primary_intervention_layer", lambda model: 28)
    return trained


def test_report_tables(report_inputs: Path) -> None:
    tables = report.model_tables(MODEL, seed=SEED, n_permutations=200)
    probes = tables["per_task"]
    assert probes.height == len(BAND) * len(TASKS)
    assert (
        probes.filter(pl.col("task") == "bbh_web_of_lies")["cot_accuracy"].to_list() == [0.75] * 2
    )
    layers = tables["layers"]
    assert layers["auroc_mean"].to_list() == pytest.approx(
        probes.group_by("layer").agg(pl.col("auroc").mean()).sort("layer")["auroc"].to_list()
    )
    assert layers["n_questions"].to_list() == [30 + 34 + 38] * 2
    comparisons = tables["layer_comparisons"]
    assert comparisons["comparison"].to_list() == ["layer 28 minus layer 26"] * 2
    by_layer = {
        layer: probes.filter(pl.col("layer") == layer).sort("task")["auroc"].to_numpy()
        for layer in BAND
    }
    auroc = comparisons.filter(pl.col("metric") == "auroc").row(0, named=True)
    assert auroc["mean_difference"] == pytest.approx((by_layer[28] - by_layer[26]).mean())
    assert auroc["n_tasks_higher"] == int((by_layer[28] > by_layer[26]).sum())
    correlations = tables["accuracy_correlations"]
    assert set(correlations["tasks"]) == {"all", "non_binary"}
    assert correlations.filter(pl.col("tasks") == "non_binary")["n_tasks"].unique().to_list() == [1]
    summary = tables["transfer_summary"]
    assert summary["off_diagonal_cells"].to_list() == [6, 6]
    assert summary["same_reasoning_type_auroc_mean"].is_not_null().all()
    assert summary["n_permutations"].to_list() == [200, 200]
    by_type = tables["transfer_by_reasoning_type"]
    assert by_type["n_cells"].sum() == 12
    stability = tables["direction_stability"]
    assert stability["weight_fold_cosine_min"].le(1.0 + 1e-9).all()
    medians = tables["direction_stability_summary"].row(0, named=True)
    assert medians["n_probes"] == len(BAND) * len(TASKS)
    assert medians["weight_fold_cosine_min_median"] == pytest.approx(
        stability["weight_fold_cosine_min"].median()
    )


def _cells(groups: dict[str, str], same: float, different: float) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source_task": a,
                "target_task": b,
                "auroc": same if groups[a] == groups[b] else different,
            }
            for a in groups
            for b in groups
            if a != b
        ]
    )


def test_the_same_type_permutation_test_shuffles_types_across_tasks() -> None:
    groups = {"a": "x", "b": "x", "c": "x", "d": "y", "e": "y", "f": "y"}
    cells = _cells(groups, 0.9, 0.5)
    test = report.same_group_test(cells, groups, seed=0, n_permutations=20_000)
    assert (test["n_same_cells"], test["n_different_cells"]) == (12, 18)
    assert test["same_auroc_mean"] == pytest.approx(0.9)
    assert test["auroc_difference"] == pytest.approx(0.4)
    # Only the observed split of the six tasks into two triples (2 of 20 label orders)
    # reaches the observed difference.
    assert test["permutation_p"] == pytest.approx(0.1, abs=0.01)
    assert report.same_group_test(cells, groups, seed=0, n_permutations=20_000) == test
    reversed_cells = _cells(groups, 0.5, 0.9)
    assert report.same_group_test(reversed_cells, groups, seed=0, n_permutations=1000)[
        "permutation_p"
    ] == pytest.approx(1.0)


def test_a_split_no_shuffle_reaches_has_the_smallest_p_value() -> None:
    groups = {f"t{i:02d}": f"g{i // 3}" for i in range(12)}
    test = report.same_group_test(_cells(groups, 0.9, 0.5), groups, seed=0, n_permutations=50)
    # One assignment in ~15,000 reproduces the observed split of 12 tasks into 4 triples.
    assert test["permutation_p"] == 1 / 51


def test_the_type_matrix_averages_cells_between_different_tasks() -> None:
    transfer = pl.DataFrame(
        {
            "model": ["m"] * 4,
            "layer": [1] * 4,
            "source_reasoning_type": ["logical", "logical", "arithmetic", "logical"],
            "target_reasoning_type": ["logical", "logical", "logical", "arithmetic"],
            "is_diagonal": [True, False, False, False],
            "auroc": [0.9, 0.7, 0.6, 0.55],
        }
    )
    table = report.transfer_by_reasoning_type(transfer)
    assert table.select("source_reasoning_type", "target_reasoning_type").rows() == [
        ("arithmetic", "logical"),
        ("logical", "arithmetic"),
        ("logical", "logical"),
    ]
    assert table["auroc_mean"].to_list() == pytest.approx([0.6, 0.55, 0.7])


def test_the_report_runner_writes_next_to_the_direction_tables(report_inputs: Path) -> None:
    trained = report_inputs
    task = OmegaConf.load(CONF / "probe_report.yaml")
    cfg = OmegaConf.create({"task": task, "model": {"model_name": MODEL}, "seed": SEED})
    tracker = probe_task.ExperimentTracker(enabled=False)
    probe_task.run_report(cfg, tracker)
    written = {p.name for p in (trained / "llama8B" / "analysis").glob("probe_*.csv")}
    assert {"probe_per_task.csv", "probe_transfer_summary.csv"} <= written

    cfg.task.output_dir = str(trained / "combined")
    probe_task.run_report(cfg, tracker)
    combined = pl.read_csv(trained / "combined" / "probe_layers.csv")
    assert combined["model"].unique().to_list() == ["llama8B"]


def test_a_missing_probe_stops_the_report_with_its_command(
    report_inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(artifacts, "TASKS", (*TASKS, "bbh_snarks"))
    with pytest.raises(FileNotFoundError, match="ri task=probe model=llama_8b") as error:
        report.require_inputs(MODEL)
    assert "bbh_snarks" in str(error.value)


def _probe_cfg(layer: int | str, skip_existing: bool) -> object:
    task = OmegaConf.load(CONF / "probe.yaml")
    task.layer, task.skip_existing = layer, skip_existing
    return OmegaConf.create(
        {
            "task": task,
            "model": {"model_name": MODEL},
            "dataset": {"name": "bbh/boolean_expressions"},
            "seed": SEED,
        }
    )


def test_the_runner_trains_a_late_band_layer(
    results_root: Path, cache: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe_task, "late_band", lambda model: BAND)
    monkeypatch.setattr(probe_task, "HS_CACHE_DIR", cache)
    tracker = probe_task.ExperimentTracker(enabled=False)

    probe_task.run_probe(_probe_cfg(28, skip_existing=False), tracker)
    assert (probe_dir(MODEL, 28, "bbh_boolean_expressions") / "metrics.json").exists()
    assert not probe_dir(MODEL, 26, "bbh_boolean_expressions").exists()
    with pytest.raises(ValueError, match="late band"):
        probe_task.run_probe(_probe_cfg(30, skip_existing=False), tracker)


def test_a_finished_layer_stops_the_band_before_any_training(
    results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained_layers: list[int] = []

    def train(model: str, dataset: str, layer: int, **kwargs: object) -> dict:
        trained_layers.append(layer)
        oof = {"auroc": 0.5, "selectivity_origin": 0.0, "selectivity_within_position": 0.0}
        return {"out_of_fold": oof, "folds": [], "full_refit": {"l2_penalty": 1.0}}

    monkeypatch.setattr(probe_task, "late_band", lambda model: BAND)
    monkeypatch.setattr(probe_task, "train_probe", train)
    finished = probe_dir(MODEL, 28, "bbh_boolean_expressions")
    finished.mkdir(parents=True)
    (finished / "metrics.json").write_text("{}")
    tracker = probe_task.ExperimentTracker(enabled=False)

    with pytest.raises(FileExistsError, match=r"task\.skip_existing=true"):
        probe_task.run_probe(_probe_cfg("band", skip_existing=False), tracker)
    assert trained_layers == []
    probe_task.run_probe(_probe_cfg("band", skip_existing=True), tracker)
    assert trained_layers == [26]


def test_an_existing_transfer_matrix_is_refused_or_skipped_before_scoring(
    results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_scoring(*args: object, **kwargs: object) -> pl.DataFrame:
        raise AssertionError("the matrix was scored")

    monkeypatch.setattr(probe_task, "late_band", lambda model: BAND)
    monkeypatch.setattr(probe_task, "evaluate_transfer", no_scoring)
    for layer in BAND:
        transfer_path(MODEL, layer).parent.mkdir(parents=True)
        transfer_path(MODEL, layer).touch()
    task = OmegaConf.load(CONF / "probe_transfer.yaml")
    cfg = OmegaConf.create({"task": task, "model": {"model_name": MODEL}, "seed": SEED})
    tracker = probe_task.ExperimentTracker(enabled=False)

    with pytest.raises(FileExistsError, match=r"task\.skip_existing=true"):
        probe_task.run_transfer(cfg, tracker)
    cfg.task.skip_existing = True
    probe_task.run_transfer(cfg, tracker)


def test_missing_probes_stop_the_transfer_runner_before_scoring(
    results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe_task, "late_band", lambda model: BAND)
    task = OmegaConf.load(CONF / "probe_transfer.yaml")
    cfg = OmegaConf.create({"task": task, "model": {"model_name": MODEL}, "seed": SEED})
    with pytest.raises(FileNotFoundError, match="ri task=probe model=llama_8b") as error:
        probe_task.run_transfer(cfg, probe_task.ExperimentTracker(enabled=False))
    assert ALL_TASKS[0] in str(error.value)


def test_task_configs_validate() -> None:
    from ri.probe.config import ProbeConfig, ProbeReportConfig, ProbeTransferConfig

    assert (
        ProbeConfig.model_validate(
            OmegaConf.to_container(OmegaConf.load(CONF / "probe.yaml"))
        ).layer
        == "band"
    )
    transfer = OmegaConf.to_container(OmegaConf.load(CONF / "probe_transfer.yaml"))
    assert ProbeTransferConfig.model_validate(transfer).layers == "band"
    probe = OmegaConf.to_container(OmegaConf.load(CONF / "probe.yaml"))
    assert transfer["skip_existing"] is probe["skip_existing"] is False
    report_cfg = OmegaConf.to_container(OmegaConf.load(CONF / "probe_report.yaml"))
    assert ProbeReportConfig.model_validate(report_cfg).output_dir is None
    with pytest.raises(ValueError):
        ProbeReportConfig.model_validate({"name": "probe_report", "out_dir": "x"})
    with pytest.raises(ValueError):
        ProbeConfig.model_validate({"name": "probe", "layer": "auto"})
