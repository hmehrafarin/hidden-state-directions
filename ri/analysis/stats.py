"""Paired tests used to compare an intervention with a prompt."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import PermutationMethod, binomtest, wilcoxon


@dataclass(frozen=True)
class McNemar:
    """A paired comparison of two boolean outcome vectors over the same questions."""

    n_pairs: int
    treated_only: int
    """Questions the treated side answers correctly and the reference does not."""

    reference_only: int
    p_value: float

    @property
    def delta(self) -> float:
        """Treated minus reference accuracy over the paired questions."""
        return (self.treated_only - self.reference_only) / self.n_pairs if self.n_pairs else 0.0


def mcnemar(treated: np.ndarray, reference: np.ndarray) -> McNemar:
    """Exact binomial McNemar test on two boolean vectors aligned question by question.

    The exact test is used because the discordant counts can be small.
    """
    treated = np.asarray(treated, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if treated.shape != reference.shape:
        raise ValueError(f"unpaired inputs: {treated.shape} vs {reference.shape}")

    b = int((treated & ~reference).sum())
    c = int((~treated & reference).sum())
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return McNemar(n_pairs=int(treated.size), treated_only=b, reference_only=c, p_value=p)


def benjamini_hochberg(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of the hypotheses that survive Benjamini-Hochberg at ``alpha``."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, p.size + 1) / p.size)
    passing = ranked <= thresholds
    keep = np.zeros(p.size, dtype=bool)
    if passing.any():
        cutoff = np.max(np.flatnonzero(passing))
        keep[order[: cutoff + 1]] = True
    return keep


@dataclass(frozen=True)
class SignedRank:
    """A two-sided Wilcoxon signed-rank test of paired per-task accuracies."""

    n: int
    """Pairs, including those with no difference."""

    n_nonzero: int
    """Pairs the test ranks: zero differences are dropped."""

    n_above: int
    n_below: int
    mean_difference: float
    statistic: float
    p_value: float
    method: str
    """How the p-value was computed; see :func:`signed_rank`."""


#: Up to this many pairs, all 2**n sign assignments are enumerated when ``exact`` cannot be used.
MAX_ENUMERATED = 13
#: Above this many pairs the normal approximation is used.
MAX_EXACT = 50


def signed_rank(treated: np.ndarray, reference: np.ndarray) -> SignedRank:
    """Two-sided Wilcoxon signed-rank test of ``treated - reference``, paired by position.

    The tabulated null distribution (``exact``) holds only without ties and zeros, which
    accuracies on tasks of a few hundred questions often produce. With them, the p-value
    enumerates every sign assignment (``permutation``) up to ``MAX_ENUMERATED`` pairs and
    uses the normal approximation (``asymptotic``) above; so does any test of more than
    ``MAX_EXACT`` pairs.
    """
    treated = np.asarray(treated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if treated.shape != reference.shape:
        raise ValueError(f"unpaired inputs: {treated.shape} vs {reference.shape}")
    difference = treated - reference
    nonzero = difference[difference != 0]
    ties = nonzero.size < difference.size or np.unique(np.abs(nonzero)).size < nonzero.size
    if difference.size > MAX_EXACT:
        method = "asymptotic"
    elif not ties:
        method = "exact"
    elif difference.size <= MAX_ENUMERATED:
        method = "permutation"
    else:
        method = "asymptotic"
    if not nonzero.size:
        statistic, p = float("nan"), 1.0
    else:
        result = wilcoxon(
            treated,
            reference,
            alternative="two-sided",
            method=PermutationMethod() if method == "permutation" else method,
        )
        statistic, p = float(result.statistic), float(result.pvalue)
    return SignedRank(
        n=int(difference.size),
        n_nonzero=int(nonzero.size),
        n_above=int((difference > 0).sum()),
        n_below=int((difference < 0).sum()),
        mean_difference=float(difference.mean()) if difference.size else float("nan"),
        statistic=statistic,
        p_value=p,
        method=method,
    )
