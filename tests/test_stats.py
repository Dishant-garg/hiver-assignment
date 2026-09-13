"""Tests for the uncertainty layer. The bootstrap is seeded, so these assert exact reproducibility
as well as correctness -- if an interval moves, `make reproduce` stops being byte-identical."""
import numpy as np
import pytest

from support_agent.eval.metrics import escalation_metrics, intent_metrics
from support_agent.eval.stats import (_macro_f1_from_codes, bootstrap_mean_ci, escalation_cost_vector,
                                      macro_f1_ci, mcnemar_exact, paired_mean_diff)


def test_escalation_cost_vector_mean_is_weighted_error():
    # The whole paired-bootstrap design rests on weighted_error being a mean of per-row costs.
    # If that ever stops holding, the intervals silently describe the wrong quantity.
    y_true = [True, True, False, False, True]
    y_pred = [True, False, True, False, False]
    c = escalation_cost_vector(y_true, y_pred, miss_cost=5.0)
    assert list(c) == [0.0, 5.0, 1.0, 0.0, 5.0]
    assert c.mean() == pytest.approx(escalation_metrics(y_true, y_pred, miss_cost=5.0)["weighted_error"])


def test_macro_f1_fast_matches_sklearn_implementation():
    # _macro_f1_from_codes duplicates intent_metrics' macro-F1 in numpy for speed; pin them together.
    labels = ["a", "b", "c"]
    y_true = ["a", "a", "b", "b", "c", "a", "b", "c"]
    y_pred = ["a", "b", "b", "c", "c", "a", "a", "b"]
    code = {l: i for i, l in enumerate(labels)}
    fast = _macro_f1_from_codes(np.array([code[v] for v in y_true]), np.array([code[v] for v in y_pred]), len(labels))
    assert fast == pytest.approx(intent_metrics(y_true, y_pred, labels)["macro_f1"])


def test_macro_f1_fast_ignores_classes_absent_from_truth():
    # intent_metrics averages over classes with support > 0 only; the fast path must agree.
    labels = ["a", "b", "c"]
    y_true, y_pred = ["a", "a", "b"], ["a", "b", "b"]
    code = {l: i for i, l in enumerate(labels)}
    fast = _macro_f1_from_codes(np.array([code[v] for v in y_true]), np.array([code[v] for v in y_pred]), len(labels))
    assert fast == pytest.approx(intent_metrics(y_true, y_pred, labels)["macro_f1"])


def test_mcnemar_counts_only_discordant_rows():
    a = [True, True, True, False, False]
    b = [True, False, False, True, False]
    r = mcnemar_exact(a, b)
    assert (r["b"], r["c"]) == (2, 1)          # a-alone-right = 2, b-alone-right = 1
    assert r["p_exact"] == pytest.approx(1.0)  # 2*P(X<=1 | n=3) = 2*(4/8) = 1.0


def test_mcnemar_identical_systems_is_p_one():
    a = [True, False, True]
    assert mcnemar_exact(a, a) == {"b": 0, "c": 0, "p_exact": 1.0}


def test_mcnemar_lopsided_split_is_significant():
    a = [True] * 20 + [False] * 2
    b = [False] * 20 + [True] * 2
    assert mcnemar_exact(a, b)["p_exact"] < 0.001


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_exact([True, False], [True])


def test_bootstrap_ci_brackets_the_point_estimate_and_is_seeded():
    x = [0.0] * 40 + [1.0] * 60
    lo, hi = bootstrap_mean_ci(x)
    assert lo < 0.6 < hi
    assert bootstrap_mean_ci(x) == [lo, hi]     # seeded: identical across calls


def test_bootstrap_ci_of_a_constant_is_degenerate():
    assert bootstrap_mean_ci([3.0] * 25) == [3.0, 3.0]


def test_bootstrap_ci_edge_cases():
    assert bootstrap_mean_ci([2.0]) == [2.0, 2.0]
    assert all(np.isnan(v) for v in bootstrap_mean_ci([]))


def test_paired_diff_is_paired_not_independent():
    # Two systems that differ by exactly 0.5 on every row have a *zero-width* paired interval,
    # even though each system's own mean is uncertain. That is the point of pairing.
    a = [1.0, 2.0, 3.0, 4.0, 5.0] * 8
    b = [0.5, 1.5, 2.5, 3.5, 4.5] * 8
    r = paired_mean_diff(a, b)
    assert r["diff"] == pytest.approx(0.5)
    assert r["ci95"] == [0.5, 0.5]
    assert r["p_a_lower"] == 0.0


def test_paired_diff_on_noise_straddles_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    r = paired_mean_diff(a, a + rng.normal(scale=1.0, size=200))
    assert r["ci95"][0] < 0 < r["ci95"][1]
    assert 0.05 < r["p_a_lower"] < 0.95


def test_paired_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_mean_diff([1.0, 2.0], [1.0])


def test_macro_f1_ci_brackets_point_estimate():
    labels = ["a", "b"]
    y_true = ["a", "b"] * 30
    y_pred = ["a", "b"] * 25 + ["b", "a"] * 5
    point = intent_metrics(y_true, y_pred, labels)["macro_f1"]
    lo, hi = macro_f1_ci(y_true, y_pred, labels, n_resamples=500)
    assert lo <= point <= hi


def test_macro_f1_ci_tolerates_unseen_predicted_label():
    # A baseline can predict a label that never appears in the golden truth; must not KeyError.
    lo, hi = macro_f1_ci(["a", "a", "b"], ["a", "zzz", "b"], ["a", "b"], n_resamples=100)
    assert 0.0 <= lo <= hi <= 1.0
