"""Judge-vs-human agreement. Quadratic-weighted kappa treats a 4-vs-5 disagreement as much
smaller than 1-vs-5, which matches how a rubric score is used."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def weighted_kappa(a: list[int], b: list[int]) -> float:
    # Kappa is undefined when a vector has no variance (sklearn returns NaN and warns); 0.0
    # ("no agreement beyond chance") is the conservative reading, so short-circuit instead of
    # calling sklearn.
    if len(set(a)) <= 1 or len(set(b)) <= 1:
        return 0.0
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    # np.corrcoef on exact rank ties (e.g. perfectly reversed ranks) can return
    # -0.9999999999999999 instead of -1.0 due to floating-point rounding in the
    # covariance/std division; round to 9 decimals and clip so a perfect (anti)
    # correlation reads as exactly +-1.0, matching what "perfect agreement" means here.
    corr = float(np.corrcoef(ra, rb)[0, 1])
    return float(max(-1.0, min(1.0, round(corr, 9))))


def agreement_report(human: list[int], judge: list[int]) -> dict:
    h, j = np.array(human), np.array(judge)
    return {"n": int(len(h)), "kappa_quadratic": weighted_kappa(list(h), list(j)), "spearman": spearman(list(h), list(j)),
            "exact_agreement": float((h == j).mean()), "within_one": float((np.abs(h - j) <= 1).mean())}
