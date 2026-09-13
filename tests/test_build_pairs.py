from pathlib import Path

import pandas as pd

from support_agent.data.build_pairs import build_pairs, clean_text, load_raw, time_split

FIXTURE = Path(__file__).parent / "fixtures" / "mini_twcs.csv"


def test_clean_text_strips_mentions_urls_and_whitespace():
    assert clean_text("@SpotifyCares my  playlist vanished https://t.co/abc") == "my playlist vanished"
    assert clean_text("@SpotifyCares @115712 charged twice!!") == "charged twice!!"
    assert clean_text("Settings &gt; Language &amp; more") == "Settings > Language & more"


def test_build_pairs_links_customer_to_brand_reply_and_turn_index():
    df = load_raw(FIXTURE)
    pairs = build_pairs(df, "SpotifyCares")
    first = pairs[pairs.customer_tweet_id == 1].iloc[0]
    assert first.brand_reply.startswith("Sorry about that!")
    assert first.turn_index == 0 and first.thread_id == 1
    second = pairs[pairs.customer_tweet_id == 3].iloc[0]
    assert second.turn_index == 1 and second.thread_id == 1


def test_build_pairs_excludes_other_brands_and_url_only_messages():
    pairs = build_pairs(load_raw(FIXTURE), "SpotifyCares")
    assert 7 not in set(pairs.customer_tweet_id)   # AppleSupport thread
    assert 9 not in set(pairs.customer_tweet_id)   # URL-only message
    assert len(pairs) == 3


def test_build_pairs_excludes_emoji_only_brand_reply():
    pairs = build_pairs(load_raw(FIXTURE), "SpotifyCares")
    assert 11 not in set(pairs.customer_tweet_id)  # brand reply cleans to "💚" (< 10 chars)
    assert len(pairs) == 3


def test_time_split_is_by_day_and_ordered():
    pairs = build_pairs(load_raw(FIXTURE), "SpotifyCares")
    corpus, holdout = time_split(pairs, frac=0.5)
    assert corpus.created_at.max() < holdout.created_at.min()
    assert len(corpus) + len(holdout) == len(pairs)
