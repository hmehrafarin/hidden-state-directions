"""Means from the activation cache, the means file, and the directions stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import ri.directions.task as directions_task
from ri.activations.store import write_activation_sample
from ri.core.naming import model_shortname
from ri.directions.means import build_means
from ri.directions.store import Means, means_path, read_means, write_means
from ri.tracking import ExperimentTracker

HID = 16
MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONF = Path(__file__).resolve().parents[1] / "ri" / "conf" / "task"


def _write(directory: Path, sample_idx: int, labels: list[bool], layers: tuple[int, ...]) -> None:
    rng = np.random.default_rng(sample_idx)
    states = {layer: rng.standard_normal((len(labels), HID)) for layer in layers}
    write_activation_sample(
        directory / f"sample_{sample_idx:04d}.h5",
        sample_idx=sample_idx,
        layer_hidden_states=states,
        layer_labels={layer: np.asarray(labels) for layer in layers},
        metadata={},
    )


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    directory = tmp_path / "cache" / model_shortname(MODEL) / "gsm8k"
    _write(directory, 0, [True, False, True, False], (22, 24))
    _write(directory, 1, [False, False, False], (22, 24))
    _write(directory, 2, [True, True], (22, 24))
    _write(directory, 4, [True, False], (22,))
    (directory / "sample_0005.h5").write_bytes(b"not hdf5")
    return directory


def test_means_are_the_plain_means_of_each_label(cache: Path) -> None:
    means = build_means(cache, 22)
    assert means.sample_idx.tolist() == [0, 1, 2, 4]
    assert means.n_plus.tolist() == [2, 0, 2, 1]
    assert means.n_minus.tolist() == [2, 3, 0, 1]

    rng = np.random.default_rng(0)
    states = rng.standard_normal((4, HID)).astype(np.float16).astype(np.float32)
    assert np.allclose(means.c_plus[0], states[[0, 2]].mean(axis=0))
    assert np.allclose(means.c_minus[0], states[[1, 3]].mean(axis=0))


def test_a_missing_side_is_nan_not_zero_and_leaves_no_direction(cache: Path) -> None:
    means = build_means(cache, 22)
    assert np.isnan(means.c_plus[1]).all()
    assert np.isnan(means.c_minus[2]).all()
    assert means.has_direction().tolist() == [True, False, False, True]


def test_questions_without_the_layer_are_left_out(cache: Path) -> None:
    assert build_means(cache, 24).sample_idx.tolist() == [0, 1, 2]


def test_an_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_means(tmp_path, 22)


def test_the_means_file_round_trips_with_fixed_dtypes(cache: Path, tmp_path: Path) -> None:
    means = build_means(cache, 22)
    path = write_means(tmp_path / "means_layer22.npz", means)
    with np.load(path) as handle:
        assert sorted(handle.files) == ["c_minus", "c_plus", "n_minus", "n_plus", "sample_idx"]
        assert handle["c_plus"].dtype == np.float32
        assert handle["sample_idx"].dtype == np.int32
        assert handle["n_plus"].dtype == np.int32
    loaded = read_means(path)
    assert np.array_equal(loaded.sample_idx, means.sample_idx)
    assert np.allclose(loaded.c_plus, means.c_plus, equal_nan=True, atol=1e-6)
    assert np.array_equal(loaded.n_minus, means.n_minus)


def test_means_reject_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="rows"):
        Means(
            sample_idx=np.arange(2),
            c_plus=np.zeros((3, 4)),
            c_minus=np.zeros((2, 4)),
            n_plus=np.zeros(2),
            n_minus=np.zeros(2),
        )


def test_the_stage_writes_one_means_file_per_layer(
    cache: Path, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(directions_task, "HS_CACHE_DIR", str(cache.parents[1]))
    monkeypatch.setattr("ri.core.layers.depth_of", lambda name: 28)
    cfg = OmegaConf.create(
        {
            "task": OmegaConf.load(CONF / "directions.yaml"),
            "model": {"model_name": MODEL},
            "dataset": {"name": "gsm8k"},
            "seed": 42,
        }
    )
    cfg.task.layers = [22, 24]
    directions_task.run(cfg, ExperimentTracker(enabled=False))

    for layer in (22, 24):
        path = means_path(MODEL, "gsm8k", layer)
        assert path == results_root / "qwen7B" / "directions" / "gsm8k" / f"means_layer{layer}.npz"
        assert np.array_equal(read_means(path).sample_idx, build_means(cache, layer).sample_idx)
