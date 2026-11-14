"""Cosines between recovering means, failed means and directions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ri.analysis.geometry import LAMBDAS, geometry_tables, means_layers, task_geometry
from ri.directions.store import Means
from ri.paths import RESULTS_ROOT_ENV, stage_dir

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _of(
    c_plus: np.ndarray,
    c_minus: np.ndarray,
    n_plus: np.ndarray | None = None,
    n_minus: np.ndarray | None = None,
) -> Means:
    n = len(c_plus)
    return Means(
        sample_idx=np.arange(n),
        c_plus=np.asarray(c_plus, dtype=np.float64),
        c_minus=np.asarray(c_minus, dtype=np.float64),
        n_plus=np.ones(n, dtype=np.int64) if n_plus is None else n_plus,
        n_minus=np.ones(n, dtype=np.int64) if n_minus is None else n_minus,
    )


def _means(n: int, d: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=d) * 3
    return {
        "sample_idx": np.arange(n),
        "c_plus": (shared + rng.normal(size=(n, d))).astype(np.float32),
        "c_minus": (shared + rng.normal(size=(n, d)) + 0.5).astype(np.float32),
        "n_plus": rng.integers(1, 5, n),
        "n_minus": rng.integers(1, 5, n),
    }


def test_the_two_means_of_a_question() -> None:
    c_plus = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    c_minus = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, -1.0]])
    row, _ = task_geometry(_of(c_plus, c_minus))
    assert row["cos_c_plus_c_minus"] == pytest.approx((np.sqrt(0.5) + 1.0 + 0.0) / 3)
    assert row["angle_c_plus_c_minus"] == pytest.approx(
        np.degrees(np.arccos(row["cos_c_plus_c_minus"]))
    )
    assert row["norm_c_plus"] == pytest.approx((1 + 2 + np.sqrt(2)) / 3)


def test_questions_without_both_means_are_left_out() -> None:
    arrays = _means(6)
    arrays["n_plus"][0] = 0
    arrays["c_plus"][0] = np.nan
    arrays["n_minus"][1] = 0
    row, _ = task_geometry(
        _of(arrays["c_plus"], arrays["c_minus"], arrays["n_plus"], arrays["n_minus"])
    )
    assert row["n_questions"] == 4
    assert all(np.isfinite(v) for v in row.values())


def test_lambda_zero_is_the_recovering_mean() -> None:
    arrays = _means(10)
    row, directions = task_geometry(
        _of(arrays["c_plus"], arrays["c_minus"], arrays["n_plus"], arrays["n_minus"])
    )
    assert row["cos_dq_between_questions_000"] == pytest.approx(row["cos_c_plus_between_questions"])
    for lam in LAMBDAS:
        assert np.linalg.norm(directions[lam]) == pytest.approx(1.0)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_identical_questions_share_the_task_level_direction() -> None:
    c_plus = np.tile([1.0, 2.0, 0.5], (4, 1))
    c_minus = np.tile([0.5, 1.0, 1.5], (4, 1))
    row, _ = task_geometry(_of(c_plus, c_minus))
    assert row["cos_dq_dT_050"] == pytest.approx(1.0)
    assert row["angle_dq_dT_050"] == pytest.approx(0.0, abs=1e-5)
    assert row["cos_dq_between_questions_050"] == pytest.approx(1.0)


def test_residual_dimensions_count_the_directions_the_residuals_spread_over() -> None:
    """Questions e0 + e_i and e0 - e_i for i = 1..4: residuals along four axes, equally."""
    eye = np.eye(8)
    c_plus = np.array([eye[0] + sign * eye[i] for i in range(1, 5) for sign in (1, -1)])
    row, directions = task_geometry(_of(c_plus, -0.5 * np.tile(eye[0], (8, 1))))
    np.testing.assert_allclose(directions[0.5], eye[0], atol=1e-12)
    assert row["residual_dims_90pct_050"] == 4
    assert row["cos_residual_between_questions_050"] == pytest.approx(-1 / 7)


def test_tables_cover_every_layer_and_task_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RESULTS_ROOT_ENV, str(tmp_path))
    for layer in (26, 28):
        for seed, task in enumerate(("gsm8k", "bbh_snarks", "bbh_navigate")):
            folder = stage_dir(MODEL, "directions", task)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder / f"means_layer{layer}.npz", **_means(12, seed=seed))
    assert means_layers(MODEL) == [26, 28]

    per_task, summary = geometry_tables(MODEL)
    assert per_task.height == 6
    assert per_task.filter(per_task["layer"] == 28)["dataset"].to_list() == [
        "gsm8k",
        "bbh_navigate",
        "bbh_snarks",
    ]
    between = summary.filter(
        (summary["layer"] == 28) & (summary["measure"] == "cos_dT_between_tasks_050")
    ).row(0, named=True)
    assert between["n"] == 6
    assert between["scope"] == "task pairs"
    mean_within = summary.filter(
        (summary["layer"] == 28) & (summary["measure"] == "cos_c_plus_c_minus")
    )["mean"].item()
    assert mean_within == pytest.approx(
        per_task.filter(per_task["layer"] == 28)["cos_c_plus_c_minus"].mean()
    )


def test_one_task_has_no_between_task_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RESULTS_ROOT_ENV, str(tmp_path))
    folder = stage_dir(MODEL, "directions", "gsm8k")
    folder.mkdir(parents=True)
    np.savez(folder / "means_layer28.npz", **_means(12))

    per_task, summary = geometry_tables(MODEL)
    assert per_task["dataset"].to_list() == ["gsm8k"]
    assert set(summary["scope"]) == {"tasks"}
    assert summary.filter(summary["measure"] == "cos_c_plus_c_minus")["n"].item() == 1
