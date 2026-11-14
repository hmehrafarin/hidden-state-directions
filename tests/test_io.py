from __future__ import annotations

import re
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from ri.utils.io import save_rows_parquet


class _Row(BaseModel):
    sample_idx: int
    answer: str | None = None


def test_rows_are_written_under_a_timestamped_name(tmp_path: Path) -> None:
    path = save_rows_parquet([_Row(sample_idx=0, answer="4"), _Row(sample_idx=1)], tmp_path / "a")
    assert re.fullmatch(r"\d{8}_\d{6}_\d{6}\.parquet", path.name)
    frame = pl.read_parquet(path)
    assert frame.to_dicts() == [
        {"sample_idx": 0, "answer": "4"},
        {"sample_idx": 1, "answer": None},
    ]
    assert list(path.parent.iterdir()) == [path]


def test_run_name_prefixes_the_file_and_mappings_are_accepted(tmp_path: Path) -> None:
    path = save_rows_parquet([{"sample_idx": 2}], tmp_path, "eval-cot")
    assert re.fullmatch(r"eval-cot_\d{8}_\d{6}_\d{6}\.parquet", path.name)
    assert pl.read_parquet(path)["sample_idx"].to_list() == [2]
