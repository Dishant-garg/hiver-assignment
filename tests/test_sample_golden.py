import pandas as pd
import pytest

from support_agent.data.sample_golden import stratified_sample


def test_stratified_sample_respects_min_per_stratum_and_size():
    rows = [{"pair_id": i, "customer_tweet_id": i, "customer_text": f"m{i}", "brand_reply": "r", "turn_index": 0,
             "created_at": "2017-11-01", "weak_label": ("rare" if i < 10 else "common")} for i in range(200)]
    df = pd.DataFrame(rows)
    s = stratified_sample(df, n=50, min_per_stratum=8, seed=1, exclude_ids=set())
    assert len(s) == 50 and (s.weak_label == "rare").sum() >= 8
    assert s.pair_id.is_unique


def test_exclude_ids_respected():
    df = pd.DataFrame([{"pair_id": i, "customer_tweet_id": i, "customer_text": f"m{i}", "brand_reply": "r", "turn_index": 0,
                        "created_at": "2017-11-01", "weak_label": "x"} for i in range(30)])
    s = stratified_sample(df, n=10, min_per_stratum=0, seed=1, exclude_ids=set(range(20)))
    assert set(s.pair_id) <= set(range(20, 30))


def test_non_first_turn_rows_never_sampled():
    rows = [{"pair_id": i, "customer_tweet_id": i, "customer_text": f"m{i}", "brand_reply": "r",
             "turn_index": (0 if i < 50 else 1), "created_at": "2017-11-01", "weak_label": "x"} for i in range(100)]
    df = pd.DataFrame(rows)
    s = stratified_sample(df, n=20, min_per_stratum=0, seed=1, exclude_ids=set())
    assert set(s.pair_id) <= set(range(50))


def test_understrength_stratum_contributes_all_rows_and_size_holds():
    rows = ([{"pair_id": i, "customer_tweet_id": i, "customer_text": f"m{i}", "brand_reply": "r", "turn_index": 0,
              "created_at": "2017-11-01", "weak_label": "rare"} for i in range(3)]
            + [{"pair_id": i, "customer_tweet_id": i, "customer_text": f"m{i}", "brand_reply": "r", "turn_index": 0,
                "created_at": "2017-11-01", "weak_label": "common"} for i in range(3, 100)])
    df = pd.DataFrame(rows)
    s = stratified_sample(df, n=20, min_per_stratum=8, seed=1, exclude_ids=set())
    assert len(s) == 20
    assert (s.weak_label == "rare").sum() == 3


def test_n_larger_than_pool_returns_whole_pool():
    df = pd.DataFrame([{"pair_id": i, "customer_tweet_id": i, "customer_text": f"m{i}", "brand_reply": "r",
                        "turn_index": 0, "created_at": "2017-11-01", "weak_label": "x"} for i in range(15)])
    s = stratified_sample(df, n=100, min_per_stratum=0, seed=1, exclude_ids=set())
    assert len(s) == 15
    assert set(s.pair_id) == set(range(15))


def test_floor_exceeding_n_raises_value_error():
    rows = [{"pair_id": i, "customer_tweet_id": i, "customer_text": f"m{i}", "brand_reply": "r", "turn_index": 0,
             "created_at": "2017-11-01", "weak_label": f"stratum{i % 5}"} for i in range(100)]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        stratified_sample(df, n=10, min_per_stratum=8, seed=1, exclude_ids=set())
