"""Uncertainty for the headline metrics.

Every number in `results/summary.md` is a point estimate on n=160. Two of the comparisons the
report makes turn on differences smaller than the sampling noise, so the point estimates on their
own invite a conclusion the data does not support. This module supplies the intervals.

Two facts make the arithmetic simple. Intent accuracy is the mean of a per-row 0/1 correctness
flag. Cost-weighted escalation error is *also* a mean -- of a per-row cost that is `miss_cost` for
a missed escalation, 1 for an unnecessary one and 0 otherwise (`escalation_cost_vector`). So both
reduce to bootstrapping a mean, and a paired comparison between two systems reduces to
bootstrapping the mean of their per-row cost *difference* -- paired, because both systems are
scored on the same rows, and ignoring that pairing would overstate the uncertainty of the gap.

Bootstrap intervals are seeded percentile intervals, so `make reproduce` regenerates them exactly.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from support_agent import config

N_RESAMPLES = 10_000


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(config.SEED if seed is None else seed)


def bootstrap_mean_ci(values: Sequence[float], *, n_resamples: int = N_RESAMPLES,
                      seed: int | None = None, alpha: float = 0.05) -> list[float]:
    """Percentile bootstrap interval for the mean of `values`.

    Vectorised: draws all resamples as one (n_resamples, n) index matrix rather than looping,
    which keeps the whole statistics pass in `run_eval` well under a second.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return [float("nan"), float("nan")]
    if n == 1:
        return [float(x[0]), float(x[0])]
    idx = _rng(seed).integers(0, n, size=(n_resamples, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


def bootstrap_ci(n: int, statistic: Callable[[np.ndarray], float], *, n_resamples: int = N_RESAMPLES,
                 seed: int | None = None, alpha: float = 0.05) -> list[float]:
    """Percentile interval for a statistic that is not a mean (macro-F1, say).

    `statistic` receives an array of row indices and returns the statistic on that resample. This
    loops, so it is reserved for statistics that genuinely need recomputation per resample.
    """
    if n == 0:
        return [float("nan"), float("nan")]
    rng = _rng(seed)
    vals = np.array([statistic(rng.integers(0, n, size=n)) for _ in range(n_resamples)], dtype=float)
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


def escalation_cost_vector(y_true: Sequence[bool], y_pred: Sequence[bool], miss_cost: float = 5.0) -> np.ndarray:
    """Per-row contribution to `escalation_metrics(...)["weighted_error"]`.

    `weighted_error` is exactly this vector's mean, which is what lets the paired comparison below
    be an ordinary mean-difference bootstrap.
    """
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    return np.where(t & ~p, miss_cost, np.where(~t & p, 1.0, 0.0))


def paired_mean_diff(a: Sequence[float], b: Sequence[float], *, n_resamples: int = N_RESAMPLES,
                     seed: int | None = None, alpha: float = 0.05) -> dict:
    """Paired bootstrap of mean(a) - mean(b), where a[i] and b[i] score the same row.

    Returns the observed difference, its interval, and `p_a_lower` -- the share of resamples in
    which a's mean came out below b's. For a cost metric, where lower is better, that is the
    bootstrap probability that system a is the better of the two. It is a directional bootstrap
    proportion, not a null-hypothesis p-value, and is labelled as such wherever it is reported.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"paired comparison needs equal-length vectors, got {len(x)} and {len(y)}")
    d = x - y
    n = len(d)
    if n == 0:
        return {"diff": float("nan"), "ci95": [float("nan"), float("nan")], "p_a_lower": float("nan"), "n": 0}
    idx = _rng(seed).integers(0, n, size=(n_resamples, n))
    diffs = d[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_a_lower": round(float((diffs < 0).mean()), 3), "n": int(n)}


def _macro_f1_from_codes(t: np.ndarray, p: np.ndarray, n_labels: int) -> float:
    """Macro-F1 over the classes present in `t`, computed straight from integer label codes.

    Deliberately duplicates `metrics.intent_metrics`' macro-F1 in pure numpy: the bootstrap needs
    it ten thousand times, and going through sklearn's `precision_recall_fscore_support` each time
    would dominate the runtime of the whole evaluation. `test_stats.py` pins the two together on
    real label vectors so the duplicate cannot drift.
    """
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    return float(f1[present].mean()) if present.any() else 0.0


def macro_f1_ci(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], *,
                n_resamples: int = N_RESAMPLES, seed: int | None = None, alpha: float = 0.05) -> list[float]:
    """Percentile interval for macro-F1. Not a mean, so it is recomputed per resample.

    Rows whose true or predicted label falls outside `labels` are dropped, which is exactly what
    `sklearn.metrics.confusion_matrix(..., labels=labels)` does inside `intent_metrics`. Matching
    it matters: if the interval were computed over a different row set than the point estimate, the
    point estimate could sit outside its own interval. A model can always emit a label that is not
    in the taxonomy, so this path is reachable in a live run, not only in tests.
    """
    code = {l: i for i, l in enumerate(labels)}
    keep = [(code[t], code[p]) for t, p in zip(y_true, y_pred) if t in code and p in code]
    if not keep:
        return [float("nan"), float("nan")]
    t = np.array([a for a, _ in keep], dtype=np.int64)
    p = np.array([b for _, b in keep], dtype=np.int64)
    return bootstrap_ci(len(t), lambda i: _macro_f1_from_codes(t[i], p[i], len(labels)),
                        n_resamples=n_resamples, seed=seed, alpha=alpha)


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Exact McNemar test that two classifiers differ, on paired per-row correctness flags.

    Only the discordant rows carry information: `b` rows where a is right and b is wrong, `c` where
    the reverse. Under the null that neither is better, each discordant row is a fair coin, so the
    exact two-sided p is twice the binomial tail at min(b, c). Computed with `math.comb` rather
    than SciPy so the package keeps its current dependency set.
    """
    a = np.asarray(correct_a, dtype=bool)
    c_ = np.asarray(correct_b, dtype=bool)
    if len(a) != len(c_):
        raise ValueError(f"McNemar needs equal-length vectors, got {len(a)} and {len(c_)}")
    b = int((a & ~c_).sum())
    c = int((~a & c_).sum())
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_exact": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b": b, "c": c, "p_exact": min(1.0, 2 * tail)}
