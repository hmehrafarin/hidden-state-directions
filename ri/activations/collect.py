"""Regenerate each question's CoT, cache its states and attach the patch-sweep labels.

The CoT is regenerated with the sweep's decoding, and its trace hashes are checked
against those stored with the sweep labels, so every cached state is paired with the
label of the same token of the same generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from ri.common import prepare_batch_data
from ri.core.model import ModelAndTokenizer
from ri.patching.pipeline import (
    TRACE_COLUMNS,
    build_cot_inputs,
    generate_cot_states,
    source_trace,
)
from ri.prompts import Prompter
from ri.tracking import ExperimentTracker
from ri.utils.extraction import extract_answer, extract_answer_from_generation
from ri.utils.tokenizer import get_eos_token_ids, get_pad_id

from .config import ActivationCollectConfig
from .store import (
    activation_sample_path,
    read_activation_labels,
    read_activation_metadata,
    sample_has_activation_layers,
    write_activation_sample,
)


@dataclass(frozen=True)
class ActivationCollectSummary:
    dataset: str
    path: Path
    samples_written: int
    samples_skipped: int


def read_patch_labels(
    path: str | Path,
    layers: list[int],
    *,
    require_trace: bool = False,
) -> pl.DataFrame:
    """Sweep labels at ``layers``, one row per (sample_idx, layer, source_position).

    Raises when the same key carries conflicting labels, or, with ``require_trace``, when
    the trace columns are missing.
    """
    required = {"sample_idx", "layer", "source_position", "correct"}
    schema = pl.read_parquet_schema(path)
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError(f"patch parquet {path} is missing required columns: {missing}")
    missing_trace = sorted(set(TRACE_COLUMNS) - set(schema))
    if require_trace and missing_trace:
        raise ValueError(
            f"strict trace validation requires patch parquet {path} to contain trace columns; "
            f"missing: {missing_trace}"
        )

    optional = {
        col for col in (*TRACE_COLUMNS, "patch_from_generation", "sweep_positions") if col in schema
    }
    df = pl.read_parquet(path, columns=sorted(required | optional))
    df = df.with_columns(
        pl.col("sample_idx").cast(pl.Int64),
        pl.col("layer").cast(pl.Int64),
        pl.col("source_position").cast(pl.Int64),
        pl.col("correct").cast(pl.Boolean),
    )
    if "patch_from_generation" in df.columns:
        df = df.filter(pl.col("patch_from_generation"))
    if "sweep_positions" in df.columns:
        df = df.filter(pl.col("sweep_positions"))
    df = df.filter((pl.col("source_position") >= 0) & pl.col("layer").is_in(layers))

    conflicts = (
        df.group_by(["sample_idx", "layer", "source_position"])
        .agg(pl.col("correct").n_unique().alias("n_unique_labels"))
        .filter(pl.col("n_unique_labels") > 1)
    )
    if not conflicts.is_empty():
        raise ValueError(f"conflicting labels for patch keys: {conflicts.head(5).to_dicts()}")

    return df.unique(["sample_idx", "layer", "source_position"], keep="last", maintain_order=True)


def _unique_non_null(values: pl.Series) -> list[Any]:
    return [value for value in values.drop_nulls().unique().to_list() if value not in ("", None)]


def _validate_trace(sample_df: pl.DataFrame, expected_trace: dict[str, Any]) -> None:
    for column, expected in expected_trace.items():
        if column not in sample_df.columns or expected is None:
            continue
        values = _unique_non_null(sample_df[column])
        if len(values) > 1:
            raise ValueError(f"patch labels contain multiple {column} values: {values[:5]}")
        if values and str(values[0]) != str(expected):
            raise ValueError(
                f"trace mismatch for {column}: patch parquet has {values[0]!r}, "
                f"regenerated CoT has {expected!r}. The CoT is regenerated as in the patch "
                "sweep, so the model, the CoT prompt and task.max_gen_len must equal the sweep's"
            )


def labels_for_sample(
    patch_df: pl.DataFrame,
    *,
    sample_idx: int,
    layers: list[int],
    num_tokens: int,
    expected_trace: dict[str, Any] | None,
) -> dict[int, np.ndarray]:
    """``{layer: [num_tokens] bool}`` for one question.

    Every position ``0..num_tokens-1`` must be labelled at every layer, and the trace
    columns, when given, must match ``expected_trace``.
    """
    sample_df = patch_df.filter(pl.col("sample_idx") == sample_idx)
    if sample_df.is_empty():
        raise ValueError(f"patch labels contain no rows for sample_idx={sample_idx}")
    if expected_trace is not None:
        _validate_trace(sample_df, expected_trace)

    expected_positions = set(range(num_tokens))
    labels: dict[int, np.ndarray] = {}
    for layer in layers:
        layer_df = sample_df.filter(pl.col("layer") == layer)
        if layer_df.is_empty():
            raise ValueError(
                f"patch labels contain no rows for sample_idx={sample_idx}, layer={layer}"
            )
        positions = [int(pos) for pos in layer_df["source_position"].to_list()]
        position_set = set(positions)
        if position_set != expected_positions:
            missing = sorted(expected_positions - position_set)[:10]
            extra = sorted(position_set - expected_positions)[:10]
            raise ValueError(
                f"source positions do not match generated token count for "
                f"sample_idx={sample_idx}, layer={layer}; missing={missing}, extra={extra}"
            )
        arr = np.zeros(num_tokens, dtype=bool)
        for pos, correct in zip(positions, layer_df["correct"].to_list(), strict=False):
            arr[int(pos)] = bool(correct)
        labels[int(layer)] = arr
    return labels


def can_resume_sample(
    path: Path,
    *,
    layers: list[int],
    patch_df: pl.DataFrame,
    sample_idx: int,
    expected_metadata: dict[str, Any],
    strict_trace: bool,
) -> bool:
    """True when ``path`` already holds this question with matching metadata and labels."""
    if not sample_has_activation_layers(path, layers, expected_metadata=expected_metadata):
        return False
    try:
        metadata = read_activation_metadata(path)
        expected_labels = labels_for_sample(
            patch_df,
            sample_idx=sample_idx,
            layers=layers,
            num_tokens=int(metadata["num_tokens"]),
            expected_trace=(
                {column: metadata.get(column) for column in TRACE_COLUMNS} if strict_trace else None
            ),
        )
        return all(
            np.array_equal(read_activation_labels(path, layer), labels)
            for layer, labels in expected_labels.items()
        )
    except (KeyError, OSError, ValueError):
        return False


class ActivationCollector:
    """Writes the activation cache of one model and one task."""

    def __init__(
        self,
        mt: ModelAndTokenizer,
        data: Any,
        *,
        dataset: str,
        prompter: Prompter,
        config: ActivationCollectConfig,
        layers: list[int],
    ) -> None:
        self.mt = mt
        self.data = data
        self.dataset = dataset
        self.prompter = prompter
        self.config = config
        self.layers = layers
        self.model_name = str(mt.model_name)

    def _generate_sample(
        self, sample_idx: int
    ) -> tuple[dict[int, np.ndarray], list[int], str, str, dict[str, Any]]:
        cfg = self.config
        _questions, answers, batched = prepare_batch_data(self.data, sample_idx, 1)
        _prompt_texts, tokenized = build_cot_inputs(self.mt, self.prompter, batched)
        gold = extract_answer([answers[0]])[0]
        pad_id = get_pad_id(self.mt.tokenizer)
        eos_ids = get_eos_token_ids(self.mt.tokenizer)

        states, generated_nested = generate_cot_states(
            self.mt,
            tokenized,
            self.layers,
            max_gen_len=cfg.max_gen_len,
            eos_ids=eos_ids,
            pad_id=pad_id,
        )
        generated_ids = generated_nested[0]
        # Cast through float32: numpy has no bfloat16; the store casts to float16.
        layer_hidden_states = {
            layer: tensor[0].detach().cpu().float().numpy() for layer, tensor in states.items()
        }
        cot_text = self.mt.tokenizer.decode(generated_ids, skip_special_tokens=True)
        source_answers = extract_answer_from_generation(
            [self.mt.tokenizer.decode(generated_ids)],
            mode=cfg.extraction_mode,
            extraction_type=cfg.answer_type,
        )
        trace = source_trace(
            self.mt,
            tokenized,
            [generated_ids],
            max_gen_len=cfg.max_gen_len,
            answer_type=cfg.answer_type,
            eos_ids=eos_ids,
            pad_id=pad_id,
        )
        flat_trace: dict[str, Any] = {key: values[0] for key, values in trace.items()}
        flat_trace["source_generated_answer"] = source_answers["final_answer"][0]

        del states
        torch.cuda.empty_cache()
        return layer_hidden_states, generated_ids, cot_text, str(gold), flat_trace

    def run(
        self,
        patch_path: str | Path,
        tracker: ExperimentTracker | None = None,
    ) -> ActivationCollectSummary:
        cfg = self.config
        patch_df = read_patch_labels(patch_path, self.layers, require_trace=cfg.strict_trace)
        identity = {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "task_code": self.dataset,
            "max_gen_len": cfg.max_gen_len,
            "seed": cfg.seed,
        }
        written = 0
        skipped = 0

        for sample_idx in tqdm(
            cfg.sample_indices(len(self.data)), desc=f"Collecting {self.dataset}"
        ):
            out_path = activation_sample_path(
                cfg.hs_cache_dir, self.model_name, self.dataset, sample_idx
            )
            if cfg.resume and can_resume_sample(
                out_path,
                layers=self.layers,
                patch_df=patch_df,
                sample_idx=sample_idx,
                expected_metadata=identity,
                strict_trace=cfg.strict_trace,
            ):
                skipped += 1
                continue

            layer_hidden_states, generated_ids, cot_text, gold, trace = self._generate_sample(
                sample_idx
            )
            labels = labels_for_sample(
                patch_df,
                sample_idx=sample_idx,
                layers=self.layers,
                num_tokens=len(generated_ids),
                expected_trace=trace if cfg.strict_trace else None,
            )
            write_activation_sample(
                out_path,
                sample_idx=sample_idx,
                layer_hidden_states=layer_hidden_states,
                layer_labels=labels,
                metadata={
                    "model_name": self.model_name,
                    "dataset": self.dataset,
                    "task_code": self.dataset,
                    "gold_answer": gold,
                    "cot_text": cot_text,
                    "num_tokens": len(generated_ids),
                    "max_gen_len": cfg.max_gen_len,
                    "seed": cfg.seed,
                    **trace,
                },
            )
            written += 1

        if tracker is not None:
            tracker.log_metrics({"samples_written": written, "samples_skipped": skipped})
        return ActivationCollectSummary(
            dataset=self.dataset,
            path=Path(cfg.hs_cache_dir),
            samples_written=written,
            samples_skipped=skipped,
        )
