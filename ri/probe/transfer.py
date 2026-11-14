"""The transfer matrix: every task's probe scored on every task, at one layer.

Cell (A, B) is the AUROC of task A's full-refit probe over every state of task B, none of
which A saw. The diagonal (B, B) uses B's out-of-fold scores, so every cell of a column
is computed on the same rows. Selectivity subtracts the AUROC of A's within-position
control probe on B's within-position labels. The origin control is reported on the
diagonal only: off it, A's arbitrary position labels are unrelated to B's.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from ri.tasks import TASKS

from .artifacts import missing_probes, probe_dir, transfer_path, write_atomic
from .data import load_task_layer
from .train import linear_scores

log = logging.getLogger(__name__)

ROW_IDENTITY = ["sample_idx", "source_position", "label", "within_position_label"]


def _source_matrix(
    model_name: str, layer: int, tasks: list[str], prefix: str
) -> tuple[np.ndarray, np.ndarray]:
    """``[H, n_tasks]`` raw weights and ``[n_tasks]`` biases of the full refits."""
    weights, biases = [], []
    for task in tasks:
        with np.load(probe_dir(model_name, layer, task) / "directions.npz") as npz:
            if int(npz["layer"]) != layer:
                raise ValueError(f"{task}: directions.npz is for layer {int(npz['layer'])}")
            weights.append(npz[f"{prefix}weight_raw"])
            biases.append(float(npz[f"{prefix}bias_raw"]))
    return np.stack(weights, axis=1), np.array(biases)


def evaluate_transfer(
    model_name: str,
    layer: int,
    *,
    hs_cache_dir: str,
    seed: int,
    device: torch.device,
    tasks: list[str] | None = None,
) -> pl.DataFrame:
    """The full matrix at one layer over ``tasks`` (all 25 by default)."""
    tasks = list(TASKS) if tasks is None else tasks
    missing = [t for t in missing_probes(model_name, layer) if t in tasks]
    if missing:
        raise RuntimeError(f"{model_name} layer {layer}: probes missing for {missing}")
    W, b = _source_matrix(model_name, layer, tasks, "")
    Wc, bc = _source_matrix(model_name, layer, tasks, "within_position_")

    records: list[dict[str, Any]] = []
    for target in tasks:
        data = load_task_layer(model_name, hs_cache_dir, target, layer, seed=seed)
        oof = pl.read_parquet(
            probe_dir(model_name, layer, target) / "out_of_fold_predictions.parquet"
        )
        if not data.rows.select(ROW_IDENTITY).equals(oof.select(ROW_IDENTITY)):
            raise RuntimeError(f"{target}: cache rows do not match the saved out-of-fold rows")
        y = oof["label"].to_numpy()
        y_control = oof["within_position_label"].to_numpy()
        real = linear_scores(data.X, W, b, device)
        control = linear_scores(data.X, Wc, bc, device)
        for a, source in enumerate(tasks):
            diagonal = source == target
            s = oof["score"].to_numpy() if diagonal else real[:, a]
            c = oof["score_within_position"].to_numpy() if diagonal else control[:, a]
            auroc = float(roc_auc_score(y, s))
            control_auroc = float(roc_auc_score(y_control, c))
            records.append(
                {
                    "source_task": source,
                    "target_task": target,
                    "is_diagonal": diagonal,
                    "n_rows": len(y),
                    "n_questions": int(oof["sample_idx"].n_unique()),
                    "positive_rate": float(y.mean()),
                    "auroc": auroc,
                    "auprc": float(average_precision_score(y, s)),
                    "within_position_control_auroc": control_auroc,
                    "selectivity_within_position": auroc - control_auroc,
                    "origin_control_auroc": float(
                        roc_auc_score(
                            oof["origin_label"].to_numpy(), oof["score_origin"].to_numpy()
                        )
                    )
                    if diagonal
                    else None,
                }
            )
        log.info("%s: %d states scored by %d source probes", target, len(y), len(tasks))
    return pl.DataFrame(records)


def summarise(matrix: pl.DataFrame) -> dict[str, Any]:
    """Diagonal against off-diagonal cells, the cells above chance, symmetry, extreme tasks."""
    diag = matrix.filter(pl.col("is_diagonal"))
    off = matrix.filter(~pl.col("is_diagonal"))
    cell = {(r["source_task"], r["target_task"]): r["auroc"] for r in off.iter_rows(named=True)}
    pairs = [(cell[(a, b)], cell[(b, a)]) for a, b in cell if a < b]
    by_source = off.group_by("source_task").agg(pl.col("auroc").mean()).sort("auroc")
    by_target = off.group_by("target_task").agg(pl.col("auroc").mean()).sort("auroc")
    return {
        "diagonal_auroc_mean": diag["auroc"].mean(),
        "off_diagonal_auroc_mean": off["auroc"].mean(),
        "off_diagonal_auroc_min": off["auroc"].min(),
        "off_diagonal_auroc_max": off["auroc"].max(),
        "off_diagonal_cells_above_chance": int((off["auroc"] > 0.5).sum()),
        "off_diagonal_cells": off.height,
        "diagonal_selectivity_within_position_mean": diag["selectivity_within_position"].mean(),
        "off_diagonal_selectivity_within_position_mean": off["selectivity_within_position"].mean(),
        "off_diagonal_cells_selective": int((off["selectivity_within_position"] > 0).sum()),
        "off_diagonal_within_position_control_mean": off["within_position_control_auroc"].mean(),
        "symmetry_spearman_rho": float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs])[0]),
        "worst_source": by_source["source_task"][0],
        "worst_source_auroc": by_source["auroc"][0],
        "best_source": by_source["source_task"][-1],
        "best_source_auroc": by_source["auroc"][-1],
        "hardest_target": by_target["target_task"][0],
        "hardest_target_auroc": by_target["auroc"][0],
        "easiest_target": by_target["target_task"][-1],
        "easiest_target_auroc": by_target["auroc"][-1],
    }


def write_transfer(model_name: str, layer: int, matrix: pl.DataFrame) -> Path:
    """Write the matrix and ``transfer_summary.json`` beside it; never replaces a matrix."""
    path = transfer_path(model_name, layer)
    if path.exists():
        raise FileExistsError(f"{path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, matrix.write_parquet)
    text = json.dumps({"model_name": model_name, "layer": layer, **summarise(matrix)}, indent=2)
    write_atomic(
        path.with_name("transfer_summary.json"), lambda p: p.write_text(text, encoding="utf-8")
    )
    return path
