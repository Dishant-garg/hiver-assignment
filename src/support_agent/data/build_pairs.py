"""Turn the flat tweet table into (customer message, brand reply) pairs for one brand.

Why pairs and not full threads: the agent answers a single incoming message, and the
brand's *first* reply to that message is the closest thing to a ground-truth resolution.

Both sides of a pair are filtered by MIN_CHARS after cleaning: a customer message that is
only a mention/URL carries no signal, and a brand reply that cleans to under MIN_CHARS
(an empty/NaN reply, or an emoji-only reply like "💚 /KT") is not a usable resolution to
learn or retrieve from.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pandas as pd

from support_agent import config

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")
MIN_CHARS = 10


def clean_text(text: str) -> str:
    # Unescape first: the raw dump stores "&gt;" / "&amp;", which would otherwise reach every
    # prompt and every retrieved precedent verbatim (e.g. "Settings &gt; Language").
    text = html.unescape(str(text))
    text = MENTION_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"tweet_id": "int64", "author_id": str, "text": str,
                                  "response_tweet_id": str, "in_response_to_tweet_id": "float64"})
    df["inbound"] = df["inbound"].astype(str).str.lower().eq("true")
    df["created_at"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    return df


def _turn_index(customer_id: int, parent_of: dict[int, int], inbound: dict[int, bool]) -> tuple[int, int]:
    """Walk up the reply chain counting customer turns; returns (turn_index, root_id)."""
    turn, cur = 0, customer_id
    while cur in parent_of:
        cur = parent_of[cur]
        if inbound.get(cur, False):
            turn += 1
    return turn, cur


def build_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    parent_of = {int(t): int(p) for t, p in zip(df.tweet_id, df.in_response_to_tweet_id) if pd.notna(p)}
    inbound = dict(zip(df.tweet_id.astype(int), df.inbound))
    text_of = dict(zip(df.tweet_id.astype(int), df.text))
    ts_of = dict(zip(df.tweet_id.astype(int), df.created_at))

    replies = df[(df.author_id == brand) & (~df.inbound) & df.in_response_to_tweet_id.notna()]
    rows, seen = [], set()
    for _, r in replies.sort_values("created_at").iterrows():
        cust_id = int(r.in_response_to_tweet_id)
        if cust_id in seen or not inbound.get(cust_id, False):
            continue  # keep only the brand's first reply to each customer tweet
        ctext = clean_text(text_of.get(cust_id, ""))
        if len(ctext) < MIN_CHARS:
            continue
        reply = clean_text(r.text)
        if len(reply) < MIN_CHARS:
            continue
        turn, root = _turn_index(cust_id, parent_of, inbound)
        seen.add(cust_id)
        rows.append({"thread_id": root, "turn_index": turn, "customer_tweet_id": cust_id,
                     "customer_text": ctext, "brand_reply": reply,
                     "created_at": ts_of.get(cust_id, r.created_at)})
    out = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    out.insert(0, "pair_id", range(len(out)))
    return out


def time_split(pairs: pd.DataFrame, frac: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by calendar day so the holdout is strictly later than the corpus (no leakage)."""
    days = sorted(pairs.created_at.dt.floor("D").unique())
    cut = days[max(0, int(len(days) * frac) - 1)]
    mask = pairs.created_at.dt.floor("D") <= cut
    return pairs[mask].reset_index(drop=True), pairs[~mask].reset_index(drop=True)


def main() -> None:
    raw = load_raw(config.RAW_DIR / "twcs.csv")
    pairs = build_pairs(raw, config.BRAND)
    corpus, holdout = time_split(pairs)
    if len(corpus) > config.CORPUS_MAX_PAIRS:
        corpus = corpus.sample(config.CORPUS_MAX_PAIRS, random_state=config.SEED).sort_values("pair_id")
    out = config.brand_dir()
    out.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(out / "corpus.csv", index=False)
    holdout.to_csv(out / "holdout.csv", index=False)
    print(f"{config.BRAND}: {len(pairs)} pairs -> corpus {len(corpus)}, holdout {len(holdout)} "
          f"(first-turn in holdout: {(holdout.turn_index == 0).sum()})", file=sys.stderr)


if __name__ == "__main__":
    main()
