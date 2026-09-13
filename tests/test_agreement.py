from support_agent.eval.agreement import agreement_report, spearman, weighted_kappa


def test_perfect_agreement():
    r = agreement_report([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert r["kappa_quadratic"] == 1.0 and r["spearman"] == 1.0 and r["exact_agreement"] == 1.0


def test_off_by_one_counts_within_one():
    r = agreement_report([1, 2, 3, 4, 5], [2, 3, 4, 5, 5])
    assert r["exact_agreement"] == 0.2 and r["within_one"] == 1.0 and r["spearman"] > 0.9


def test_reverse_is_negative():
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == -1.0
    assert weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0


def test_zero_variance_kappa_is_zero_without_warning():
    assert weighted_kappa([3, 3, 3], [3, 3, 3]) == 0.0
    assert agreement_report([2, 2, 2, 2], [2, 2, 3, 2])["kappa_quadratic"] == 0.0
