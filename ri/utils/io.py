"""Writing result rows to Parquet."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel


def write_parquet_atomic(frame: pl.DataFrame, path: Path) -> Path:
    """Write ``frame`` to ``path`` through a process-unique temporary file and a rename.

    A reader never sees a partial file, and the temporary name does not end in
    ``.parquet``, so a glob for result files never picks it up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)
    return path


def save_rows_parquet(
    rows: Sequence[BaseModel | Mapping[str, Any]],
    output_dir: str | Path,
    run_name: str | None = None,
) -> Path:
    """Write ``rows`` to ``<output_dir>/<run_name>_<YYYYMMDD_HHMMSS_ffffff>.parquet``."""
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{run_name}_{ts}" if run_name else ts
    records = [row.model_dump() if isinstance(row, BaseModel) else dict(row) for row in rows]
    return write_parquet_atomic(pl.DataFrame(records), Path(output_dir) / f"{stem}.parquet")
