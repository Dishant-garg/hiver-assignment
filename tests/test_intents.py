from support_agent import intents


def test_taxonomy_shape():
    assert 8 <= len(intents.INTENTS) <= 10
    assert intents.INTENTS[-1].name == intents.OTHER
    assert len(set(intents.INTENT_NAMES)) == len(intents.INTENT_NAMES)
    for i in intents.INTENTS[:-1]:
        assert len(i.examples) >= 2 and i.definition.endswith(".")
    assert hash(intents.INTENTS[0])  # frozen dataclass must really be immutable/hashable


def test_other_is_never_auto_handled():
    assert intents.by_name(intents.OTHER).auto_handle_allowed is False
    assert intents.by_name("nonexistent").name == intents.OTHER


def test_weak_label_uses_keywords():
    assert intents.weak_label("songs keep skipping and it will not play") == "playback_issue"
    assert intents.weak_label("my account was hacked and I cannot log in") == "account_access"
    assert intents.weak_label("you charged me twice, I want a refund") == "billing_and_refund"
    assert intents.weak_label("all my saved songs are gone") == "library_playlist_issue"
    assert intents.weak_label("") == intents.OTHER
    assert intents.weak_label("good morning") == intents.OTHER


def test_weak_label_is_robust_to_curly_quotes_and_non_strings():
    assert intents.weak_label("it won’t play and it won’t let me listen") == "playback_issue"
    assert intents.weak_label(None) == intents.OTHER
    assert intents.weak_label(float("nan")) == intents.OTHER


def test_weak_label_counts_nested_keywords_once():
    """"double charged" must not also score "charged"; one piece of evidence, one vote."""
    hits = intents._distinct_hits(intents._PATTERNS["billing_and_refund"], "double charged")
    assert hits == 1
    assert intents._distinct_hits(intents._PATTERNS["billing_and_refund"], "charged and billed") == 2


def test_intents_needing_account_access_are_not_auto_handled():
    """The escalation gate reads this flag; anything touching an account or money must escalate."""
    escalate = {"account_access", "billing_and_refund", "subscription_plan", intents.OTHER}
    for name in escalate:
        assert intents.by_name(name).auto_handle_allowed is False


def test_definitions_block_is_compact_and_complete():
    block = intents.definitions_block()
    for name in intents.INTENT_NAMES:
        assert f"- {name}:" in block
    # Lands in every classifier prompt. Bound raised from 1100 to 1150 when the close-pair
    # discriminators ("plan change -> subscription_plan") were added to the definitions.
    assert len(block) < 1150
