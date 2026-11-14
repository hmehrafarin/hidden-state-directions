"""The linear probe estimator: an L2-penalised binary cross-entropy minimised to convergence.

The probe predicts a binary label through ``sigmoid(w . x + b)`` and minimises

    mean binary cross-entropy + (l2_penalty / 2) * ||w||^2        (bias unpenalised)

This objective is strictly convex, so it has one minimiser, reached from any start; it
equals scikit-learn's L2-penalised linear classifier with ``C = 1 / (l2_penalty * n)``.
Full-batch L-BFGS with a strong-Wolfe line search runs in float64 so that the line
search does not stall on rounding before ``max |gradient| <= grad_tol`` is met.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ProbeFit:
    """A fitted probe in the coordinates it was fitted in, and how the solve ended."""

    weight: torch.Tensor
    """``[H]`` float64, on the fitting device."""
    bias: torch.Tensor
    """``[]`` float64, on the fitting device."""
    l2_penalty: float
    iterations: int
    converged: bool
    grad_max: float
    objective: float


def fit_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    l2_penalty: float,
    *,
    max_iter: int,
    grad_tol: float,
    init: ProbeFit | None = None,
) -> ProbeFit:
    """Minimise mean BCE + (l2_penalty / 2)||w||^2 over ``(w, b)`` to convergence.

    Args:
        X: ``[n, H]`` float64 features.
        y: ``[n]`` float64 labels in {0, 1}.
        l2_penalty: must be positive; without it separable data has no minimiser.
        max_iter: L-BFGS iteration cap; the evaluation cap is twice this.
        grad_tol: converged when every gradient coordinate is at most this in magnitude.
        init: warm start, e.g. the fit at the neighbouring penalty on the same rows. The
            minimiser is unique, so this changes the iteration count, not the answer.
    """
    if X.dtype != torch.float64 or y.dtype != torch.float64:
        raise TypeError(f"fit in float64: got X {X.dtype}, y {y.dtype}")
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError(f"shapes X {tuple(X.shape)}, y {tuple(y.shape)}")
    if l2_penalty <= 0:
        raise ValueError(f"l2_penalty must be positive, got {l2_penalty}")

    if init is None:
        weight = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
        bias = torch.zeros((), dtype=torch.float64, device=X.device)
    else:
        weight = init.weight.detach().clone()
        bias = init.bias.detach().clone()
    weight.requires_grad_(True)
    bias.requires_grad_(True)

    opt = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=max_iter,
        max_eval=2 * max_iter,
        tolerance_grad=grad_tol,
        tolerance_change=0.0,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(X @ weight + bias, y)
        loss = loss + 0.5 * l2_penalty * weight.dot(weight)
        loss.backward()
        return loss

    opt.step(closure)
    objective = float(closure())
    grad_weight = cast(torch.Tensor, weight.grad)
    grad_bias = cast(torch.Tensor, bias.grad)
    grad_max = max(float(grad_weight.abs().max()), float(grad_bias.abs()))
    return ProbeFit(
        weight=weight.detach(),
        bias=bias.detach(),
        l2_penalty=float(l2_penalty),
        iterations=int(opt.state[weight]["n_iter"]),
        converged=grad_max <= grad_tol,
        grad_max=grad_max,
        objective=objective,
    )
