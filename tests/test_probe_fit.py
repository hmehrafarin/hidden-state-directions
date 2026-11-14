"""The probe solver and the question folds."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from ri.probe.artifacts import assign_folds, probe_dir, transfer_path
from ri.probe.fit import fit_probe


def _reference_minimiser(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """``(w, b)`` minimising mean BCE + (l2 / 2)||w||^2, by scipy's L-BFGS-B."""

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = theta[:-1], theta[-1]
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * l2 * w @ w
        grad = np.append(X.T @ (p - y) / len(y) + l2 * w, np.mean(p - y))
        return loss, grad

    theta0 = np.zeros(X.shape[1] + 1)
    return minimize(objective, theta0, jac=True, method="L-BFGS-B", tol=1e-14).x


def test_the_fit_reaches_the_penalised_minimiser() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 30))
    y = (X[:, :3].sum(axis=1) + rng.standard_normal(600) > 0).astype(float)
    for l2 in (1e-2, 1e-1, 1.0):
        fit = fit_probe(torch.tensor(X), torch.tensor(y), l2, max_iter=10_000, grad_tol=1e-8)
        reference = _reference_minimiser(X, y, l2)
        assert fit.converged
        assert fit.grad_max <= 1e-8
        np.testing.assert_allclose(fit.weight.numpy(), reference[:-1], rtol=1e-4, atol=1e-6)
        assert float(fit.bias) == pytest.approx(reference[-1], abs=1e-5)


def test_a_warm_start_reaches_the_same_minimiser() -> None:
    rng = np.random.default_rng(1)
    X = torch.tensor(rng.standard_normal((400, 20)))
    y = torch.tensor((rng.random(400) < 0.5).astype(float))
    strong = fit_probe(X, y, 1.0, max_iter=10_000, grad_tol=1e-8)
    cold = fit_probe(X, y, 1e-2, max_iter=10_000, grad_tol=1e-8)
    warm = fit_probe(X, y, 1e-2, max_iter=10_000, grad_tol=1e-8, init=strong)
    assert strong.converged and cold.converged and warm.converged
    torch.testing.assert_close(warm.weight, cold.weight, rtol=1e-5, atol=1e-7)


def test_the_fit_refuses_single_precision_and_a_zero_penalty() -> None:
    X = torch.zeros((4, 2), dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    with pytest.raises(TypeError):
        fit_probe(X.float(), y, 1.0, max_iter=10, grad_tol=1e-6)
    with pytest.raises(ValueError, match="positive"):
        fit_probe(X, y, 0.0, max_iter=10, grad_tol=1e-6)


def test_folds_depend_on_the_task_name_and_its_questions_only() -> None:
    folds = assign_folds("bbh_navigate", range(250))
    assert folds == assign_folds("bbh/navigate", reversed(range(250)))
    sizes = np.bincount(list(folds.values()))
    assert sizes.max() - sizes.min() <= 1
    assert folds != assign_folds("bbh_snarks", range(250))


def test_folds_are_the_paper_assignment() -> None:
    folds = assign_folds("gsm8k", range(20))
    assert "".join(str(folds[q]) for q in range(20)) == "31121231024403443020"


def test_too_few_questions_cannot_fill_the_folds() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        assign_folds("gsm8k", range(4))


def test_artifact_locations(results_root) -> None:
    model = "meta-llama/Llama-3.1-8B-Instruct"
    assert probe_dir(model, 28, "bbh/snarks") == results_root / "llama8B/probe/layer28/bbh_snarks"
    assert transfer_path(model, 28) == results_root / "llama8B/probe/layer28/transfer.parquet"
