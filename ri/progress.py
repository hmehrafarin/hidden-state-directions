"""Resumable progress of the patch sweep, safe for concurrent jobs on a shared filesystem.

Completed ``(sample_idx, layer)`` pairs are read from the combined parquet, and
:meth:`ProgressTracker.absorb` merges a per-question working file into it and deletes
the source. All reads and writes hold an ``flock`` on a sibling ``.lock`` file, and
writes go through a process-unique temporary file and an atomic rename.
"""

from __future__ import annotations

import fcntl
import logging
from pathlib import Path

import polars as pl

from ri.utils.io import write_parquet_atomic

log = logging.getLogger(__name__)

# Columns kept in the combined parquet; the prompt columns are too heavy to keep.
KEEP_COLUMNS = [
    "sample_idx",
    "layer",
    "source_position",
    "source_token",
    "source_generation_id",
    "source_num_tokens",
    "source_token_ids_hash",
    "source_prompt_hash",
    "patch_from_generation",
    "sweep_positions",
    "gold_final_answer",
    "generated_final_answer",
    "correct",
    "generated_text",
]

PATCH_REQUIRED_COLUMNS = [
    "sample_idx",
    "layer",
    "source_position",
    "source_generation_id",
    "source_num_tokens",
    "source_token_ids_hash",
]


class ProgressTracker:
    """Completed ``(sample_idx, layer)`` pairs of one combined patch-sweep parquet.

    A pair counts as done only if its source-trace metadata is consistent and every
    source position is present. Every rejected pair is logged with the reason.
    """

    def __init__(self, combined_path: str | Path) -> None:
        self._combined_path = Path(combined_path)
        self._combined_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._combined_path.with_suffix(".lock")
        self._done: set[tuple[int, int]] = set()
        self.invalid_reasons: list[str] = []
        self._load_done()
        for reason in self.invalid_reasons:
            log.warning("not resumable, will be recomputed: %s", reason)

    def _load_done(self) -> None:
        with open(self._lock_path, "a+") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            try:
                self.invalid_reasons = []
                self._done = set()
                if self._combined_path.exists():
                    self._done = self._valid_patch_pairs(pl.read_parquet(self._combined_path))
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        log.info(
            "%d validated completed (sample, layer) pairs in %s",
            len(self._done),
            self._combined_path.name,
        )

    def is_done(self, *, sample_idx: int, layer: int) -> bool:
        """Whether the ``(sample_idx, layer)`` pair is complete."""
        return (sample_idx, layer) in self._done

    def absorb(self, parquet_path: str | Path) -> None:
        """Merge ``parquet_path`` into the combined file, then delete it.

        Rows of the combined file that share a ``(sample_idx, layer)`` pair with the new
        file are replaced as a whole group.
        """
        parquet_path = Path(parquet_path)
        new_df = pl.read_parquet(parquet_path)
        new_df = new_df.select([c for c in KEEP_COLUMNS if c in new_df.columns])

        with open(self._lock_path, "a+") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                if self._combined_path.exists():
                    existing = pl.read_parquet(self._combined_path)
                    keys = {"sample_idx", "layer"}
                    if keys.issubset(new_df.columns) and keys.issubset(existing.columns):
                        pairs = new_df.select("sample_idx", "layer").unique()
                        existing = existing.join(pairs, on=["sample_idx", "layer"], how="anti")
                    combined = pl.concat([existing, new_df], how="diagonal_relaxed")
                else:
                    combined = new_df
                write_parquet_atomic(combined, self._combined_path)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

        parquet_path.unlink()
        self._load_done()

        log.info(
            "Absorbed %s (%d rows) into %s | total pairs: %d",
            parquet_path.name,
            len(new_df),
            self._combined_path.name,
            len(self._done),
        )

    def _valid_patch_pairs(self, df: pl.DataFrame) -> set[tuple[int, int]]:
        missing = [col for col in PATCH_REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            self.invalid_reasons.append(
                f"{self._combined_path.name}: missing source trace metadata columns {missing}; "
                "existing rows will not be trusted for resume"
            )
            return set()

        valid: set[tuple[int, int]] = set()
        metadata_by_pair: dict[tuple[int, int], str] = {}
        for keys, group in df.group_by(["sample_idx", "layer"], maintain_order=True):
            sample_idx, layer = int(keys[0]), int(keys[1])
            pair = (sample_idx, layer)
            lengths = [
                int(v)
                for v in group["source_num_tokens"].drop_nulls().unique().to_list()
                if int(v) > 0
            ]
            generation_ids = [
                str(v)
                for v in group["source_generation_id"].drop_nulls().unique().to_list()
                if str(v)
            ]
            token_hashes = [
                str(v)
                for v in group["source_token_ids_hash"].drop_nulls().unique().to_list()
                if str(v)
            ]
            if len(lengths) != 1 or len(generation_ids) != 1 or len(token_hashes) != 1:
                self.invalid_reasons.append(
                    f"{self._combined_path.name}: inconsistent source trace metadata "
                    f"for sample_idx={sample_idx}, layer={layer}"
                )
                continue

            expected_len = lengths[0]
            positions = [int(v) for v in group["source_position"].drop_nulls().to_list()]
            if len(positions) != expected_len or set(positions) != set(range(expected_len)):
                self.invalid_reasons.append(
                    f"{self._combined_path.name}: incomplete source positions for "
                    f"sample_idx={sample_idx}, layer={layer}; expected 0..{expected_len - 1}"
                )
                continue
            valid.add(pair)
            metadata_by_pair[pair] = generation_ids[0]

        generation_by_sample: dict[int, str] = {}
        conflicting_samples: set[int] = set()
        for (sample_idx, _layer), generation_id in metadata_by_pair.items():
            prev = generation_by_sample.setdefault(sample_idx, generation_id)
            if prev != generation_id:
                conflicting_samples.add(sample_idx)

        if conflicting_samples:
            valid = {pair for pair in valid if pair[0] not in conflicting_samples}
            for sample_idx in sorted(conflicting_samples):
                self.invalid_reasons.append(
                    f"{self._combined_path.name}: source generation differs across layers "
                    f"for sample_idx={sample_idx}; rows will not be trusted for resume"
                )
        return valid
