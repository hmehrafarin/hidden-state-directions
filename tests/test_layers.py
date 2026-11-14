from __future__ import annotations

import json
from pathlib import Path

import pytest
from huggingface_hub import constants

from ri import settings
from ri.core.layers import (
    DEFAULT_BAND_SIZE,
    depth_of,
    late_band,
    num_hidden_layers,
    primary_intervention_layer,
    resolve_layers,
)

LLAMA_BLOCKS = 32
QWEN_BLOCKS = 28


def test_late_band_of_the_two_models() -> None:
    assert late_band(LLAMA_BLOCKS) == [26, 28, 30]
    assert late_band(QWEN_BLOCKS) == [22, 24, 26]


def test_primary_intervention_layer_is_the_middle_of_the_band() -> None:
    assert primary_intervention_layer(LLAMA_BLOCKS) == 28
    assert primary_intervention_layer(QWEN_BLOCKS) == 24
    for blocks in (LLAMA_BLOCKS, QWEN_BLOCKS, 20, 24, 40, 80):
        assert primary_intervention_layer(blocks) in late_band(blocks)


def test_late_band_excludes_the_final_layer() -> None:
    for blocks in (LLAMA_BLOCKS, QWEN_BLOCKS, 7, 24, 40, 80):
        assert blocks - 1 not in late_band(blocks, 2)


def test_odd_block_count() -> None:
    assert late_band(7, 2) == [2, 4]


def test_late_band_rejects_a_band_that_does_not_fit() -> None:
    with pytest.raises(ValueError, match="band size"):
        late_band(LLAMA_BLOCKS, 0)
    with pytest.raises(ValueError, match="candidate layers"):
        late_band(4, DEFAULT_BAND_SIZE)


def test_resolve_band() -> None:
    assert resolve_layers("band", QWEN_BLOCKS) == [22, 24, 26]
    assert resolve_layers("band", QWEN_BLOCKS, band_size=2) == [24, 26]


def test_resolve_explicit_list_resolves_negative_indices() -> None:
    assert resolve_layers([0, -1], LLAMA_BLOCKS) == [0, 31]
    assert resolve_layers([0, -1], QWEN_BLOCKS) == [0, 27]


def test_resolve_rejects_layers_beyond_model_depth() -> None:
    for bad in ([28], [30], [26, 28, 30]):
        with pytest.raises(ValueError, match="invalid for a model"):
            resolve_layers(bad, QWEN_BLOCKS)
    assert resolve_layers([26, 28, 30], LLAMA_BLOCKS) == [26, 28, 30]


def test_resolve_rejects_unknown_spec() -> None:
    for spec in ("auto", "middle"):
        with pytest.raises(ValueError, match="unknown layer spec"):
            resolve_layers(spec, LLAMA_BLOCKS)


def test_num_hidden_layers_accepts_int_config_or_model() -> None:
    class _Config:
        num_hidden_layers = QWEN_BLOCKS

    class _Model:
        config = _Config()

    assert num_hidden_layers(QWEN_BLOCKS) == QWEN_BLOCKS
    assert num_hidden_layers(_Config()) == QWEN_BLOCKS
    assert num_hidden_layers(_Model()) == QWEN_BLOCKS
    with pytest.raises(ValueError):
        num_hidden_layers(True)
    with pytest.raises(ValueError):
        num_hidden_layers(object())


def test_num_hidden_layers_of_a_real_model(tiny_causal_lm: object) -> None:
    assert num_hidden_layers(tiny_causal_lm) == 6
    assert late_band(tiny_causal_lm, 2) == [2, 4]


def test_depth_of_reads_a_local_model_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"num_hidden_layers": QWEN_BLOCKS}))
    assert depth_of(str(model_dir)) == QWEN_BLOCKS
    assert resolve_layers("band", str(model_dir)) == [22, 24, 26]

    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        depth_of(str(tmp_path / "empty"))


def test_depth_of_finds_a_model_in_the_standard_hub_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With RI_CACHE_DIR unset, a config downloaded by the Hugging Face tools is found."""
    snapshot = tmp_path / "models--org--Cached-Model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"num_hidden_layers": LLAMA_BLOCKS}))
    (tmp_path / "models--org--Cached-Model" / "refs").mkdir()
    (tmp_path / "models--org--Cached-Model" / "refs" / "main").write_text("abc123")
    monkeypatch.setattr(settings, "HUGGINGFACE_CACHE_DIR", None)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    depth_of.cache_clear()
    assert depth_of("org/Cached-Model") == LLAMA_BLOCKS
    with pytest.raises(ValueError, match="local model directory"):
        depth_of("org/Missing-Model")
