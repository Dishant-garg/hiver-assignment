"""The golden set is the evaluation ground truth, so its shape and label contract are tested
like code: a typo in an intent name or a should_escalate/escalation_reason pair that
contradicts itself would silently corrupt every metric downstream."""
import pandas as pd

from support_agent import config, intents, rules

COLUMNS = ["golden_id", "pair_id", "customer_tweet_id", "customer_text", "reference_reply",
           "intent", "should_escalate", "escalation_reason", "notes"]


def _load(name):
    return pd.read_csv(config.GOLDEN_DIR / name)


def test_golden_shape_and_labels():
    g = _load("golden_set.csv")
    assert 150 <= len(g) <= 250 and g.golden_id.is_unique and g.pair_id.is_unique
    assert g.intent.isin(intents.INTENT_NAMES).all()
    assert g.escalation_reason.isin(rules.ESCALATION_CATEGORIES).all()
    assert ((g.escalation_reason == "none") == (~g.should_escalate)).all()
    assert g.intent.nunique() >= len(intents.INTENTS) - 1


def test_dev_disjoint_from_golden():
    g, d = _load("golden_set.csv"), _load("dev_set.csv")
    assert len(d) >= 30 and not set(g.pair_id) & set(d.pair_id)


def test_both_sets_have_the_documented_columns_and_no_empty_cells():
    """Every column the eval reads, and a note on every row: the labelling guide requires a
    note on non-obvious calls and the labeller wrote one everywhere, so an empty note means a
    row was merged without a label rather than that the call was easy."""
    for name in ("golden_set.csv", "dev_set.csv"):
        df = _load(name)
        assert list(df.columns) == COLUMNS, name
        assert df.notna().all().all(), name
        assert df.escalation_reason.isin(rules.ESCALATION_CATEGORIES).all(), name
        assert ((df.escalation_reason == "none") == (~df.should_escalate)).all(), name


# --- Audit invariants -------------------------------------------------------------------
# These encode the audit that was run over the 160 labels by hand. They are tests rather than a
# paragraph in a document because "I checked" is not evidence a reviewer can re-run.

def _corpus():
    return pd.read_csv(config.brand_dir() / "corpus.csv")


def test_golden_never_leaks_into_the_retrieval_corpus():
    """The agent is scored on messages it could not have retrieved. Leakage on any of the three
    keys would let the retriever hand the model the answer it is being graded against."""
    g, c = _load("golden_set.csv"), _corpus()
    assert not set(g.pair_id) & set(c.pair_id)
    assert not set(g.customer_tweet_id) & set(c.customer_tweet_id)
    assert not set(g.customer_text) & set(c.customer_text)


def test_golden_is_strictly_later_than_the_corpus():
    """Split by calendar day, so no thread and no same-day boilerplate straddles the line."""
    g, c = _load("golden_set.csv"), _corpus()
    hold = pd.read_csv(config.brand_dir() / "holdout.csv")
    rows = hold[hold.pair_id.isin(g.pair_id)]
    assert len(rows) == len(g), "every golden row must come from the held-out slice"
    assert (rows.turn_index == 0).all(), "golden rows are first turns only"
    assert c.created_at.max() < rows.created_at.min()


def test_no_duplicate_messages_within_the_golden_set():
    g = _load("golden_set.csv")
    assert not g.customer_text.duplicated().any()
    assert not g.customer_tweet_id.duplicated().any()


def test_every_never_auto_handle_intent_escalates():
    """Guide (b) rules 5-7: needs_account_access covers account_access and subscription_plan,
    billing_dispute covers billing_and_refund, and `other` is never auto-handled. So a row
    carrying one of those intents and should_escalate=False is a contradiction, and it would
    silently teach the escalation metric that such rows are safe to automate."""
    never_auto = [i.name for i in intents.INTENTS if not i.auto_handle_allowed]
    for name in ("golden_set.csv", "dev_set.csv"):
        df = _load(name)
        offenders = df[df.intent.isin(never_auto) & ~df.should_escalate.astype(bool)]
        assert offenders.empty, f"{name}: {list(offenders.golden_id)}"


def test_pre_check_agrees_with_the_hand_labels_where_it_fires():
    """Every row `pre_check` fires on must be a row the labeller escalated, and for the same
    reason. This is the check that caught the three guide/implementation mismatches in decision
    log 12: before that fix it fired on two rows and agreed with the recorded reason on neither."""
    g = _load("golden_set.csv")
    fired = disagreed = 0
    for _, r in g.iterrows():
        hits = [h.rule for h in rules.pre_check(str(r.customer_text))]
        if not hits:
            continue
        fired += 1
        assert bool(r.should_escalate), f"golden {r.golden_id}: rules fire {hits} but label says auto-handle"
        if r.escalation_reason not in hits:
            disagreed += 1
    assert fired >= 8, f"pre_check fires on only {fired} golden rows; the DM/phishing patterns regressed"
    assert disagreed == 0, f"{disagreed} rows where the rule's reason contradicts the hand-recorded one"
