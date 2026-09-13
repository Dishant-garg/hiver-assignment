from support_agent.eval.metrics import confusion_markdown, escalation_metrics, intent_metrics


def test_intent_metrics_hand_computed():
    m = intent_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert m["accuracy"] == 0.75
    assert abs(m["per_class"]["a"]["f1"] - 2 / 3) < 1e-9
    assert m["confusion"] == [[1, 1], [0, 2]]
    assert "| a |" in confusion_markdown(m)


def test_escalation_metrics_hand_computed():
    # truth: 2 escalate, 2 auto. pred: one missed escalation, one unnecessary.
    m = escalation_metrics([True, True, False, False], [True, False, True, False], miss_cost=5.0)
    assert m["recall"] == 0.5 and m["precision"] == 0.5
    assert m["missed_escalation_rate"] == 0.25 and m["unnecessary_escalation_rate"] == 0.25
    assert m["weighted_error"] == (5.0 * 1 + 1.0 * 1) / 4
    assert m["automation_rate"] == 0.5
