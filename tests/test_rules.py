from support_agent import rules


def hits(msg):
    return {h.rule for h in rules.pre_check(msg)}


def test_human_request_detected():
    assert "human_requested" in hits("can I talk to a real person please")
    assert "human_requested" in hits("I want to speak to an agent")


def test_asking_to_be_dmd_is_a_human_request():
    """Guide (b) rule 4 counts "to be phoned/DM'd by a human" as an explicit human request.

    This is the most common phrasing in the golden set and the rule missed all of it, which is
    most of why the rules-only baseline had 0.026 recall."""
    for msg in ["Hi please see my DM :)", "Can you please answer my DM? Thanks",
                "I have an artist issue. DM me plz", "can we talk in DM?", "please call me"]:
        assert "human_requested" in hits(msg), msg


def test_legal_and_safety_detected():
    assert "safety_legal" in hits("I will sue you, my lawyer is ready")
    assert "safety_legal" in hits("this is a scam and fraud")


def test_pii_detected():
    assert "pii" in hits("my card 4111 1111 1111 1111 was charged")
    assert "pii" in hits("email me at john.doe@example.com")


def test_abusive_detected_and_clean_message_has_no_hits():
    assert "abusive" in hits("you are all f***ing useless")
    assert "abusive" in hits("fuck you")
    assert hits("my playlist disappeared after the update") == set()


def test_swearing_about_the_product_is_not_abuse():
    """Guide (b) rule 2 carve-out, verbatim: "Frustrated swearing *about* the product ('this damn
    app') is not abuse on its own; abuse aimed at a person is." The rule used to match bare
    profanity and so flagged the guide's own counter-example."""
    for msg in ["Your web player fucking blows.", "this is shit",
                "does not give me permission to log in cause i deactivated Facebook account . WTF"]:
        assert "abusive" not in hits(msg), msg


def test_phishing_and_discrimination_are_safety_legal():
    """Guide (b) rule 1 names fraud and discrimination; neither phishing nor sexism matched before."""
    assert "safety_legal" in hits("received email asking me to follow a link. is this real or phishing?")
    assert "safety_legal" in hits("there's no need to use lazy casual sexism in your ads")


def test_post_check_thresholds():
    from support_agent import intents
    ok = next(i.name for i in intents.INTENTS if i.auto_handle_allowed)
    kw = dict(conf_threshold=0.6, retr_threshold=0.15, max_chars=280)
    assert {h.rule for h in rules.post_check(ok, 0.3, 0.5, "ok", **kw)} == {"low_confidence"}
    assert {h.rule for h in rules.post_check(ok, 0.9, 0.05, "ok", **kw)} == {"weak_evidence"}
    assert {h.rule for h in rules.post_check("other", 0.9, 0.5, "ok", **kw)} == {"intent_not_auto_handleable"}
    assert {h.rule for h in rules.post_check(ok, 0.9, 0.5, "You will get a full refund of $30 guaranteed", **kw)} == {"forbidden_promise"}
    assert {h.rule for h in rules.post_check(ok, 0.9, 0.5, "x" * 300, **kw)} == {"too_long"}


def test_routine_complaints_are_not_abusive_or_pii():
    assert "abusive" not in hits("the app is useless right now")
    assert "abusive" not in hits("this update broke everything, such garbage design")
    assert "abusive" in hits("you guys are useless")
    assert "pii" not in hits("order number 1234567890123 never arrived")
    assert "pii" not in hits("case 5551234567 still open")
    assert "pii" in hits("my card is 4111-1111-1111-1111")
    assert "pii" in hits("call me at +1 415-555-0134")


def test_timing_language_is_only_a_promise_in_money_context():
    kw = dict(conf_threshold=0.0, retr_threshold=0.0, max_chars=280)
    from support_agent import intents
    ok = next(i.name for i in intents.INTENTS if i.auto_handle_allowed)
    assert rules.post_check(ok, 0.9, 0.5, "We'll follow up within 24 hours with an update.", **kw) == []
    assert {h.rule for h in rules.post_check(ok, 0.9, 0.5, "Your refund will land within 3 days.", **kw)} == {"forbidden_promise"}
    assert {h.rule for h in rules.post_check(ok, 0.9, 0.5, "We guarantee this is fixed.", **kw)} == {"forbidden_promise"}
