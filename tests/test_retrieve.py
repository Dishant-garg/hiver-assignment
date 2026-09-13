import pandas as pd

from support_agent.retrieve import Retriever


def corpus():
    return pd.DataFrame({
        "pair_id": [0, 1, 2],
        "customer_text": ["my playlist disappeared after the update",
                          "charged twice for premium this month",
                          "songs keep skipping every few seconds"],
        "brand_reply": ["Try logging out and back in.", "Let's check that charge.", "Try clearing cache."],
    })


def test_top_k_returns_best_match_first():
    r = Retriever(corpus())
    ev = r.top_k("why was I charged twice", k=2)
    assert ev[0].pair_id == 1 and ev[0].brand_reply == "Let's check that charge."
    assert ev[0].score >= ev[1].score and 0 <= ev[0].score <= 1


def test_no_overlap_scores_zero():
    r = Retriever(corpus())
    assert r.top_k("zzzz qqqq", k=1)[0].score == 0.0


def test_tied_scores_break_by_corpus_order_not_by_sort_luck():
    """Identical cosine scores must resolve identically on every machine.

    np.argsort defaults to an unstable quicksort, so tied neighbours came back in an order that
    depended on the BLAS underneath. That reordered the prompt and so changed the LLM cache key:
    the committed cache replayed on the machine that recorded it and missed in CI. Three identical
    documents are the sharpest version of the case -- every score is exactly equal, so corpus
    order is the only thing left to rank them by.
    """
    import pandas as pd

    from support_agent.retrieve import Retriever

    corpus = pd.DataFrame({"pair_id": [10, 11, 12, 13],
                           "customer_text": ["same text here", "same text here", "same text here", "different"],
                           "brand_reply": ["a", "b", "c", "d"]})
    got = [e.pair_id for e in Retriever(corpus).top_k("same text here", k=3)]
    assert got == [10, 11, 12], got


def test_ranking_survives_a_perturbation_larger_than_any_blas_difference():
    """Scores nudged well below _TIE_PRECISION must not reorder the results.

    This is what actually differed across platforms: not the ranking logic but the low-order bits
    of the cosine scores feeding it, which land around 1e-16 for a sparse dot product. The
    perturbation here is 1e-9 -- far larger than that, still far smaller than the rounding -- so
    passing means real BLAS variation cannot reach the ranking.
    """
    import numpy as np
    import pandas as pd

    from support_agent import retrieve as retrieve_mod
    from support_agent.retrieve import Retriever

    corpus = pd.DataFrame({"pair_id": [1, 2, 3, 4],
                           "customer_text": ["alpha beta", "alpha beta", "alpha beta gamma", "delta"],
                           "brand_reply": ["w", "x", "y", "z"]})
    r = Retriever(corpus)
    baseline = [e.pair_id for e in r.top_k("alpha beta", k=3)]

    real = retrieve_mod.cosine_similarity
    rng = np.random.default_rng(0)
    try:
        retrieve_mod.cosine_similarity = lambda a, b: real(a, b) + rng.normal(0, 1e-9, real(a, b).shape)
        perturbed = [e.pair_id for e in r.top_k("alpha beta", k=3)]
    finally:
        retrieve_mod.cosine_similarity = real
    assert perturbed == baseline, f"{perturbed} != {baseline}"
