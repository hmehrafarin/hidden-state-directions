from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from ri.progress import ProgressTracker


def _rows(
    positions: list[int],
    *,
    sample_idx: int = 0,
    layer: int = 7,
    generation_id: str = "gen-a",
    token_hash: str = "tok-a",
    n: int = 3,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_idx": sample_idx,
            "layer": layer,
            "source_position": pos,
            "source_generation_id": generation_id,
            "source_num_tokens": n,
            "source_token_ids_hash": token_hash,
            "source_token": f"tok{pos}",
            "patch_from_generation": True,
            "sweep_positions": True,
            "gold_final_answer": "A",
            "generated_final_answer": "A",
            "correct": True,
            "generated_text": "",
        }
        for pos in positions
    ]


def test_progress_accepts_complete_source_position_group(tmp_path: Path) -> None:
    path = tmp_path / "combined.parquet"
    pl.DataFrame(_rows([0, 1, 2])).write_parquet(path)

    tracker = ProgressTracker(path)

    assert tracker.is_done(sample_idx=0, layer=7)
    assert tracker.invalid_reasons == []


def test_progress_rejects_incomplete_source_position_group(tmp_path: Path) -> None:
    path = tmp_path / "combined.parquet"
    pl.DataFrame(_rows([0, 2])).write_parquet(path)

    tracker = ProgressTracker(path)
    assert not tracker.is_done(sample_idx=0, layer=7)
    assert "incomplete source positions" in tracker.invalid_reasons[0]


def test_progress_rejects_cross_layer_source_generation_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "combined.parquet"
    rows = _rows([0, 1], layer=0, generation_id="gen-a", token_hash="tok-a", n=2)
    rows += _rows([0, 1], layer=2, generation_id="gen-b", token_hash="tok-b", n=2)
    pl.DataFrame(rows).write_parquet(path)

    tracker = ProgressTracker(path)
    assert not tracker.is_done(sample_idx=0, layer=0)
    assert not tracker.is_done(sample_idx=0, layer=2)
    assert "source generation differs across layers" in tracker.invalid_reasons[-1]


def test_progress_rejects_missing_trace_columns(tmp_path: Path) -> None:
    path = tmp_path / "combined.parquet"
    pl.DataFrame({"sample_idx": [0], "layer": [7], "source_position": [0]}).write_parquet(path)
    tracker = ProgressTracker(path)
    assert not tracker.is_done(sample_idx=0, layer=7)
    assert "missing source trace metadata" in tracker.invalid_reasons[0]


def test_absorb_replaces_entire_stale_sample_layer_group(tmp_path: Path) -> None:
    combined = tmp_path / "combined.parquet"
    stale_rows = [
        {
            "sample_idx": 0,
            "layer": 7,
            "source_position": pos,
            "source_token": f"old{pos}",
            "patch_from_generation": True,
            "gold_final_answer": "A",
            "generated_final_answer": "A",
            "correct": True,
            "generated_text": "",
        }
        for pos in range(4)
    ]
    pl.DataFrame(stale_rows).write_parquet(combined)

    new_path = tmp_path / "new.parquet"
    pl.DataFrame(_rows([0, 1], n=2)).write_parquet(new_path)

    tracker = ProgressTracker(combined)
    tracker.absorb(new_path)

    df = pl.read_parquet(combined)
    assert sorted(df["source_position"].to_list()) == [0, 1]
    assert df["source_token"].to_list() == ["tok0", "tok1"]
    assert tracker.is_done(sample_idx=0, layer=7)
    assert not new_path.exists()


def test_absorb_drops_heavy_columns(tmp_path: Path) -> None:
    new_path = tmp_path / "new.parquet"
    rows = [{**row, "target_prompt": "long"} for row in _rows([0, 1, 2])]
    pl.DataFrame(rows).write_parquet(new_path)
    combined = tmp_path / "combined.parquet"
    ProgressTracker(combined).absorb(new_path)
    assert "target_prompt" not in pl.read_parquet(combined).columns


def test_rejected_pairs_are_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "combined.parquet"
    pl.DataFrame(_rows([0, 2])).write_parquet(path)
    with caplog.at_level(logging.WARNING, logger="ri.progress"):
        ProgressTracker(path)
    assert "incomplete source positions for sample_idx=0, layer=7" in caplog.text
