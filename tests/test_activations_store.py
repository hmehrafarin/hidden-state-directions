"""The HDF5 activation cache: layout, file format and Hub-backed caches."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ri.activations import store

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


def _write(path: Path, labels: np.ndarray, layers: tuple[int, ...] = (0, 2)) -> np.ndarray:
    rng = np.random.default_rng(0)
    states = rng.standard_normal((len(labels), 8)).astype(np.float32)
    store.write_activation_sample(
        path,
        sample_idx=3,
        layer_hidden_states={layer: states for layer in layers},
        layer_labels={layer: labels for layer in layers},
        metadata={"model_name": LLAMA, "dataset": "gsm8k", "num_tokens": len(labels), "x": None},
    )
    return states


def test_local_cache_layout(tmp_path: Path) -> None:
    path = store.activation_sample_path(tmp_path / "cache", LLAMA, "bbh_snarks", 7)
    assert path == tmp_path / "cache" / "llama8B" / "bbh_snarks" / "sample_0007.h5"
    assert not (tmp_path / "cache").exists(), "resolving a path creates nothing"


def test_file_format(tmp_path: Path) -> None:
    """Datasets, dtypes and compression of the cache file format."""
    labels = np.array([True, False, True])
    path = tmp_path / "sample_0003.h5"
    states = _write(path, labels)

    with h5py.File(path) as handle:
        assert int(handle.attrs["sample_idx"]) == 3
        assert "x" not in handle.attrs
        assert sorted(handle["layers"]) == ["0", "2"]
        group = handle["layers/2"]
        assert sorted(group) == ["hidden_states", "labels", "source_positions"]
        assert group["hidden_states"].dtype == np.float16
        assert group["labels"].dtype == np.bool_
        assert group["source_positions"].dtype == np.int32
        for name in group:
            assert group[name].compression == "gzip"
            assert group[name].compression_opts == 9
            assert group[name].shuffle
        np.testing.assert_array_equal(group["source_positions"][()], [0, 1, 2])

    np.testing.assert_array_equal(store.read_activation_labels(path, 2), labels)
    read = store.read_activation_hidden_states(path, 0)
    assert read.dtype == np.float32
    np.testing.assert_array_equal(read, states.astype(np.float16).astype(np.float32))
    assert store.read_activation_layer_ids(path) == [0, 2]
    assert store.read_activation_metadata(path)["dataset"] == "gsm8k"
    assert not path.with_suffix(".h5.tmp").exists()


def test_write_rejects_mismatched_layers_and_lengths(tmp_path: Path) -> None:
    states = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="same layers"):
        store.write_activation_sample(
            tmp_path / "a.h5",
            sample_idx=0,
            layer_hidden_states={0: states},
            layer_labels={1: np.zeros(2, dtype=bool)},
            metadata={},
        )
    with pytest.raises(ValueError, match="labels"):
        store.write_activation_sample(
            tmp_path / "b.h5",
            sample_idx=0,
            layer_hidden_states={0: states},
            layer_labels={0: np.zeros(3, dtype=bool)},
            metadata={},
        )


def test_has_layers_checks_layers_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample_0003.h5"
    assert not store.sample_has_activation_layers(path, [0])
    _write(path, np.array([True, False]))
    assert store.sample_has_activation_layers(path, [0, 2], expected_metadata={"num_tokens": 2})
    assert store.sample_has_activation_layers(path, [0], expected_metadata={"dataset": None})
    assert not store.sample_has_activation_layers(path, [0, 4])
    assert not store.sample_has_activation_layers(path, [0], expected_metadata={"dataset": "x"})


def test_hf_cache_is_downloaded_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, tuple[str, tuple[str, ...]]] = {}

    def fake_snapshot(repo_id: str, patterns: tuple[str, ...]) -> Path:
        calls["args"] = (repo_id, patterns)
        return tmp_path / "snap"

    monkeypatch.setattr(store, "_hf_snapshot", fake_snapshot)
    (tmp_path / "snap" / "llama8B" / "gsm8k").mkdir(parents=True)

    got = store.activation_sample_dir("hf://datasets/user/repo", LLAMA, "gsm8k")
    assert got == tmp_path / "snap" / "llama8B" / "gsm8k"
    assert calls["args"] == ("user/repo", ("llama8B/gsm8k/*",))

    with pytest.raises(FileNotFoundError):
        store.activation_sample_dir("hf://user/repo", LLAMA, "strategyqa")
