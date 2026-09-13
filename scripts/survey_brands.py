"""One-off survey that produced the brand decision. Prints a markdown table of the
top brands by customer->brand reply pairs, reply length, deflection share (DM/link
hand-offs, broadly defined), resolution-step share (actual troubleshooting language),
and customer non-English share (a lexical retriever needs mostly-English traffic)."""
import sys

import pandas as pd

from support_agent import config

# Narrow "DM us" phrasing only (kept for comparison with the first survey pass).
DM_PATTERN = r"\b(?:dm|direct message|send us a (?:private )?message)\b"

# Broader deflection: any hand-off out of the thread (DM, "reach out", "contact us",
# a channel switch, or a bare link) rather than an in-thread resolution.
DEFLECTION_PATTERN = (
    r"\b(?:dm|direct message|send us a (?:private )?message|reach out|contact us|"
    r"get in touch|email us|call us|chat with us)\b|https?://"
)

# Resolution-step language: words that show up when a reply actually walks the
# customer through a fix, rather than just apologizing or deflecting.
RESOLUTION_PATTERN = (
    r"\b(?:try|restart|reinstall|re-install|update|clear|log ?out|sign ?in|sign ?out|"
    r"steps?|check|reset|refresh|settings?|enable|disable|toggle|uninstall|reboot|"
    r"delete|remove|troubleshoot)\b"
)


def main(top: int = 12) -> None:
    df = pd.read_csv(config.RAW_DIR / "twcs.csv", usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"])
    brands = df.loc[~df.inbound, "author_id"].value_counts().head(top).index
    rows = []
    by_id = df.set_index("tweet_id")
    for b in brands:
        replies = df[(df.author_id == b) & (~df.inbound)]
        parent_ids = replies.in_response_to_tweet_id.dropna().astype(int)
        parent_inbound = by_id.reindex(parent_ids).inbound.fillna(False)
        n_pairs = int(parent_inbound.sum())

        dm_share = replies.text.str.contains(DM_PATTERN, regex=True, case=False).mean()
        deflection_share = replies.text.str.contains(DEFLECTION_PATTERN, regex=True, case=False).mean()
        resolution_share = replies.text.str.contains(RESOLUTION_PATTERN, regex=True, case=False).mean()
        med_len = replies.text.str.len().median()

        # Unique customer (inbound) tweets this brand replied to, to measure how much
        # of the brand's incoming traffic looks non-English (a lexical/TF-IDF retriever
        # needs mostly-English traffic to be useful).
        parent_lookup = by_id.reindex(parent_ids.unique())
        inbound_parent_ids = parent_lookup.index[parent_lookup.inbound.fillna(False)]
        customer_texts = by_id.loc[inbound_parent_ids, "text"]
        non_ascii_counts = customer_texts.str.count(r"[^\x00-\x7F]")
        non_english_share = (non_ascii_counts > 5).mean()

        rows.append({
            "brand": b,
            "reply_tweets": len(replies),
            "customer_pairs": n_pairs,
            "median_reply_chars": med_len,
            "dm_boilerplate_share": round(dm_share, 3),
            "deflection_any_share": round(deflection_share, 3),
            "resolution_step_share": round(resolution_share, 3),
            "customer_non_english_share": round(non_english_share, 3),
        })
    out = pd.DataFrame(rows).sort_values("customer_pairs", ascending=False)
    print(out.to_markdown(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
