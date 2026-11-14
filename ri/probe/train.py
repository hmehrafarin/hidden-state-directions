"""Cross-fitted training of one linear probe per (task, layer).

The probe reads one CoT state and predicts whether patching it into the direct prompt
recovers the correct answer.

* **Cross-fitting.** For each of the five question folds, a probe fitted on the other
  four scores the fold. Pooled, these out-of-fold scores give one AUROC (area under the
  ROC curve) over every state of every question of the task.
* **Nested penalty choice.** Fold k's L2 penalty is chosen by leave-one-fold-out
  cross-validation over its four training folds, so fold k never informs its own probe.
  A penalty is eligible only if every fit at it converged (``max|grad| <= GRAD_TOL``
  within ``MAX_ITER`` L-BFGS iterations).
* **Full refit.** One more probe is fitted on every question, its penalty chosen by
  5-fold cross-validation over all questions. It is the source probe for transfer.
* **Controls** (Hewitt & Liang, 2019): the origin control (one arbitrary label per source
  position) and the within-position control (real labels shuffled within position and
  fold), cross-fitted at the real probe's penalties. Selectivity is AUROC minus control
  AUROC.

Training rows are class-balanced within each fold, by downsampling whichever class is
the majority there, and standardised with the training
rows' own mean and standard deviation; scoring uses every row. Probes are saved in raw
hidden-state units, ``w / sd`` with the standardisation folded into the bias.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import N_FOLDS, probe_dir, write_atomic
from .data import TaskLayerData, load_task_layer
from .fit import ProbeFit, fit_probe

log = logging.getLogger(__name__)

L2_GRID: tuple[float, ...] = (1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4)
"""Strongest first: each fit on a training set warm-starts the next, weaker one."""
MAX_ITER = 10_000
GRAD_TOL = 1e-6
SCORE_CHUNK = 32_768
LABEL_COLUMNS = {
    "real": "label",
    "origin": "origin_label",
    "within_position": "within_position_label",
}
CONTROLS = ("origin", "within_position")


@dataclass(frozen=True)
class RawProbe:
    """A probe in raw hidden-state units: ``score = x . weight + bias``."""

    weight: np.ndarray
    bias: float
    l2_penalty: float
    converged: bool
    iterations: int


@dataclass
class _TrainingSet:
    Z: torch.Tensor
    mean: torch.Tensor
    sd: torch.Tensor
    pool_rows: np.ndarray


class _CrossFitter:
    """Fits on unions of folds' pool rows and scores whole folds, on one device."""

    def __init__(self, data: TaskLayerData, device: torch.device) -> None:
        self.data = data
        self.device = device
        self.fold = data.rows["fold"].to_numpy().astype(np.int64)
        self.labels = {
            name: data.rows[col].to_numpy().astype(np.float64)
            for name, col in LABEL_COLUMNS.items()
        }
        self.pool_idx = np.flatnonzero(data.rows["in_pool"].to_numpy())
        self.pool_fold = self.fold[self.pool_idx]
        self.X_pool = torch.from_numpy(data.X[self.pool_idx]).to(device)
        self.fits = 0

    def training_set(self, train_folds: list[int]) -> _TrainingSet:
        """The pool rows of ``train_folds``, standardised with their own mean and sd."""
        # Built in blocks so that the float64 matrix is the only full-size allocation.
        selected = np.isin(self.pool_fold, train_folds)
        idx = torch.from_numpy(np.flatnonzero(selected)).to(self.device)
        Z = torch.empty((len(idx), self.X_pool.shape[1]), dtype=torch.float64, device=self.device)
        for start in range(0, len(idx), SCORE_CHUNK):
            rows = idx[start : start + SCORE_CHUNK]
            Z[start : start + len(rows)] = self.X_pool.index_select(0, rows)
        mean = Z.mean(dim=0)
        sq = torch.zeros_like(mean)
        for start in range(0, len(Z), SCORE_CHUNK):
            sq += (Z[start : start + SCORE_CHUNK] - mean).square().sum(dim=0)
        sd = (sq / len(Z)).sqrt()
        sd = torch.where(sd == 0, torch.ones_like(sd), sd)
        Z.sub_(mean).div_(sd)
        return _TrainingSet(Z=Z, mean=mean, sd=sd, pool_rows=self.pool_idx[selected])

    def release(self) -> None:
        """Return cached GPU blocks: training sets differ in size, which fragments the cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def fit_path(self, ts: _TrainingSet, label: str, l2_values: list[float]) -> list[RawProbe]:
        """One converged fit per penalty, strongest first, each warm-starting the next."""
        y = torch.from_numpy(self.labels[label][ts.pool_rows]).to(self.device)
        probes: list[RawProbe] = []
        init: ProbeFit | None = None
        for l2 in l2_values:
            fit = fit_probe(ts.Z, y, l2, max_iter=MAX_ITER, grad_tol=GRAD_TOL, init=init)
            self.fits += 1
            init = fit
            weight = fit.weight / ts.sd
            probes.append(
                RawProbe(
                    weight=weight.cpu().numpy(),
                    bias=float(fit.bias - (ts.mean * weight).sum()),
                    l2_penalty=float(l2),
                    converged=fit.converged,
                    iterations=fit.iterations,
                )
            )
        return probes

    def encoding(self, ts: _TrainingSet, probe: RawProbe) -> dict[str, np.ndarray]:
        """The activation pattern ``Cov(x) w`` and the class-mean difference, in raw units.

        The filter ``w`` of a discriminative probe partly cancels nuisance covariance;
        the pattern (Haufe et al., 2014) and the class-mean difference describe where
        recovering states lie. Both are computed on the probe's own training rows.
        """
        weight_z = torch.from_numpy(probe.weight).to(self.device) * ts.sd
        y = torch.from_numpy(self.labels["real"][ts.pool_rows]).to(self.device)
        cov_w = ts.Z.T @ (ts.Z @ weight_z) / len(ts.Z)
        diff = ts.Z.T @ (y / y.sum() - (1 - y) / (1 - y).sum())
        return {
            "pattern": (ts.sd * cov_w).cpu().numpy(),
            "mean_difference": (ts.sd * diff).cpu().numpy(),
        }

    def score(self, rows: np.ndarray, probes: list[RawProbe]) -> np.ndarray:
        """``[len(rows), len(probes)]`` raw-space logits, in float64."""
        weights = np.stack([p.weight for p in probes], axis=1)
        biases = np.array([p.bias for p in probes])
        return linear_scores(self.data.X, weights, biases, self.device, rows)


def linear_scores(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    device: torch.device,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """``X[rows] @ W + b`` in float64, ``[n_rows, k]`` for ``W`` of shape ``[H, k]``.

    Computed in blocks of rows, so that only one block of ``X`` is ever held in float64.
    """
    Wt = torch.from_numpy(W).to(device, dtype=torch.float64)
    bt = torch.from_numpy(b).to(device, dtype=torch.float64)
    index = np.arange(len(X)) if rows is None else rows
    out = np.empty((len(index), W.shape[1]))
    for start in range(0, len(index), SCORE_CHUNK):
        chunk = index[start : start + SCORE_CHUNK]
        xb = torch.from_numpy(X[chunk]).to(device, dtype=torch.float64)
        out[start : start + len(chunk)] = (xb @ Wt + bt).cpu().numpy()
    return out


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    if y.min() == y.max():
        raise ValueError("AUROC is undefined on a single class")
    return float(roc_auc_score(y, score))


def cross_fit(
    data: TaskLayerData, *, device: torch.device, l2_grid: tuple[float, ...] = L2_GRID
) -> dict[str, Any]:
    """Nested cross-fitting of the real probe and both controls, plus the full refit."""
    grid = list(l2_grid)
    if grid != sorted(grid, reverse=True) or len(set(grid)) != len(grid):
        raise ValueError(f"l2_grid must be strictly descending: {grid}")
    fitter = _CrossFitter(data, device)
    folds = list(range(N_FOLDS))
    fold_rows = {k: np.flatnonzero(fitter.fold == k) for k in folds}
    y_real = fitter.labels["real"]

    oof = {name: np.full(len(fitter.fold), np.nan) for name in LABEL_COLUMNS}
    outer: list[dict[str, Any]] = []
    fold_probes: dict[str, list[RawProbe]] = {name: [] for name in LABEL_COLUMNS}
    top_level = np.zeros((N_FOLDS, len(grid)))
    top_level_converged = np.ones((N_FOLDS, len(grid)), dtype=bool)
    fold_encodings: list[dict[str, np.ndarray]] = []

    for k in folds:
        inner = np.zeros((N_FOLDS - 1, len(grid)))
        inner_converged = np.ones(len(grid), dtype=bool)
        for i, j in enumerate(f for f in folds if f != k):
            ts = fitter.training_set([f for f in folds if f not in (k, j)])
            probes = fitter.fit_path(ts, "real", grid)
            del ts
            fitter.release()
            scores = fitter.score(fold_rows[j], probes)
            inner[i] = [_auroc(y_real[fold_rows[j]], scores[:, g]) for g in range(len(grid))]
            inner_converged &= np.array([p.converged for p in probes])
        if not inner_converged.any():
            raise RuntimeError(f"fold {k}: no penalty converged on every inner fit")
        inner_mean = inner.mean(axis=0)
        g_k = int(np.argmax(np.where(inner_converged, inner_mean, -np.inf)))

        # The whole path on the four training folds also gives the full refit's CV curve;
        # the fit at g_k is fold k's probe.
        ts = fitter.training_set([f for f in folds if f != k])
        probes = fitter.fit_path(ts, "real", grid)
        scores = fitter.score(fold_rows[k], probes)
        top_level[k] = [_auroc(y_real[fold_rows[k]], scores[:, g]) for g in range(len(grid))]
        top_level_converged[k] = [p.converged for p in probes]
        if not probes[g_k].converged:
            raise RuntimeError(f"fold {k}: the probe at l2={grid[g_k]} did not converge")
        oof["real"][fold_rows[k]] = scores[:, g_k]
        fold_probes["real"].append(probes[g_k])
        fold_encodings.append(fitter.encoding(ts, probes[g_k]))
        for control in CONTROLS:
            (probe,) = fitter.fit_path(ts, control, [grid[g_k]])
            if not probe.converged:
                raise RuntimeError(f"fold {k}: {control} control did not converge")
            oof[control][fold_rows[k]] = fitter.score(fold_rows[k], [probe])[:, 0]
            fold_probes[control].append(probe)
        del ts
        fitter.release()
        outer.append(
            {
                "fold": k,
                "n_questions": int(data.rows.filter(pl.col("fold") == k)["sample_idx"].n_unique()),
                "n_rows": len(fold_rows[k]),
                "l2_penalty": grid[g_k],
                "l2_at_grid_edge": g_k in (0, len(grid) - 1),
                "inner_cv_auroc": dict(zip(map(str, grid), inner_mean.tolist(), strict=True)),
                "inner_converged": dict(zip(map(str, grid), inner_converged.tolist(), strict=True)),
                "auroc": _auroc(y_real[fold_rows[k]], oof["real"][fold_rows[k]]),
                "iterations": probes[g_k].iterations,
            }
        )

    # Not nested: the full refit is only ever scored on other tasks.
    curve = top_level.mean(axis=0)
    usable = cast(np.ndarray, top_level_converged.all(axis=0))
    if not usable.any():
        raise RuntimeError("no penalty converged on every top-level fit")
    g_full = int(np.argmax(np.where(usable, curve, -np.inf)))
    ts = fitter.training_set(folds)
    full = {name: fitter.fit_path(ts, name, [grid[g_full]])[0] for name in LABEL_COLUMNS}
    full_encoding = fitter.encoding(ts, full["real"])
    del ts
    fitter.release()
    for name, probe in full.items():
        if not probe.converged:
            raise RuntimeError(f"full refit ({name}) did not converge at l2={grid[g_full]}")

    if np.isnan(oof["real"]).any():
        raise RuntimeError("some rows received no out-of-fold score")
    auroc = _auroc(y_real, oof["real"])
    fold_aurocs = np.array([o["auroc"] for o in outer])
    origin = _auroc(fitter.labels["origin"], oof["origin"])
    within_position = _auroc(fitter.labels["within_position"], oof["within_position"])
    return {
        "oof": oof,
        "fold_probes": fold_probes,
        "full": full,
        "encodings": {"folds": fold_encodings, "full": full_encoding},
        "summary": {
            "out_of_fold": {
                "auroc": auroc,
                "auprc": float(average_precision_score(y_real, oof["real"])),
                "auroc_fold_mean": float(fold_aurocs.mean()),
                "auroc_fold_sd": float(fold_aurocs.std(ddof=1)),
                "control_origin_auroc": origin,
                "control_within_position_auroc": within_position,
                "selectivity_origin": auroc - origin,
                "selectivity_within_position": auroc - within_position,
            },
            "folds": outer,
            "full_refit": {
                "l2_penalty": grid[g_full],
                "l2_at_grid_edge": g_full in (0, len(grid) - 1),
                "cv_auroc": dict(zip(map(str, grid), curve.tolist(), strict=True)),
                "cv_converged": dict(zip(map(str, grid), usable.tolist(), strict=True)),
                "iterations": full["real"].iterations,
            },
            "n_fits": fitter.fits,
            "n_pool_rows": len(fitter.pool_idx),
        },
    }


def write_artifacts(
    out_dir: Path,
    model_name: str,
    data: TaskLayerData,
    result: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Save the probes, the out-of-fold scores and the metrics; ``metrics.json`` goes last.

    ``directions.npz`` holds the full refit and the five fold probes in raw units, their
    activation patterns and class-mean differences, and the control probes.
    """
    if (out_dir / "metrics.json").exists():
        raise FileExistsError(f"{out_dir} already holds a finished probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    full, folds = result["full"], result["fold_probes"]
    encodings = result["encodings"]
    arrays: dict[str, Any] = {
        "layer": np.int64(data.layer),
        "weight_raw": full["real"].weight,
        "bias_raw": np.float64(full["real"].bias),
        "l2_penalty": np.float64(full["real"].l2_penalty),
        "fold_weight_raw": np.stack([p.weight for p in folds["real"]]),
        "fold_bias_raw": np.array([p.bias for p in folds["real"]]),
        "fold_l2_penalty": np.array([p.l2_penalty for p in folds["real"]]),
        "pattern_raw": encodings["full"]["pattern"],
        "mean_difference_raw": encodings["full"]["mean_difference"],
        "fold_pattern_raw": np.stack([e["pattern"] for e in encodings["folds"]]),
        "fold_mean_difference_raw": np.stack([e["mean_difference"] for e in encodings["folds"]]),
    }
    for control in CONTROLS:
        arrays[f"{control}_weight_raw"] = full[control].weight
        arrays[f"{control}_bias_raw"] = np.float64(full[control].bias)
        arrays[f"fold_{control}_weight_raw"] = np.stack([p.weight for p in folds[control]])
        arrays[f"fold_{control}_bias_raw"] = np.array([p.bias for p in folds[control]])
    np.savez(out_dir / "directions.npz", **arrays)

    data.rows.select(pl.lit(data.dataset).alias("dataset"), "sample_idx", "fold").unique().sort(
        "sample_idx"
    ).write_parquet(out_dir / "question_folds.parquet")
    data.rows.with_columns(
        pl.Series("score", result["oof"]["real"].astype(np.float32)),
        pl.Series("score_origin", result["oof"]["origin"].astype(np.float32)),
        pl.Series("score_within_position", result["oof"]["within_position"].astype(np.float32)),
    ).write_parquet(out_dir / "out_of_fold_predictions.parquet")

    metrics = {
        "model_name": model_name,
        "dataset": data.dataset,
        "layer": data.layer,
        "n_questions": int(data.rows["sample_idx"].n_unique()),
        "n_rows": int(data.rows.height),
        "positive_rate": float(cast(float, data.rows["label"].mean())),
        **result["summary"],
        **meta,
        "written_at": datetime.now(UTC).isoformat(),
    }
    text = json.dumps(metrics, indent=2, sort_keys=True)
    write_atomic(out_dir / "metrics.json", lambda path: path.write_text(text, encoding="utf-8"))
    return metrics


def train_probe(
    model_name: str,
    dataset: str,
    layer: int,
    *,
    hs_cache_dir: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train, cross-fit and save one (task, layer) probe; return its metrics."""
    start = time.time()
    data = load_task_layer(model_name, hs_cache_dir, dataset, layer, seed=seed)
    loaded = time.time()
    log.info(
        "%s layer %d: %d states of %d questions, hidden size %d",
        data.dataset,
        layer,
        data.rows.height,
        data.rows["sample_idx"].n_unique(),
        data.X.shape[1],
    )
    result = cross_fit(data, device=device)
    meta = {
        "n_folds": N_FOLDS,
        "seed": seed,
        "l2_grid": list(L2_GRID),
        "lbfgs_max_iter": MAX_ITER,
        "lbfgs_grad_tol": GRAD_TOL,
        "seconds_loading": round(loaded - start, 1),
        "seconds_fitting": round(time.time() - loaded, 1),
        "hidden_size": int(data.X.shape[1]),
        "device": device.type,
    }
    return write_artifacts(probe_dir(model_name, layer, dataset), model_name, data, result, meta)
