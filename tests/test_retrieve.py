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
