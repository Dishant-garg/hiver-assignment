import json

import pandas as pd

from support_agent import intents
from support_agent.eval.run import run_eval
from support_agent.llm import LLMClient


def test_run_eval_end_to_end(tmp_path, monkeypatch):
    auto_ok = next(i.name for i in intents.INTENTS if i.auto_handle_allowed)
    golden = pd.DataFrame({"golden_id": [0, 1], "pair_id": [100, 101], "customer_tweet_id": [1, 2],
                           "customer_text": ["my playlist disappeared", "I will sue you"],
                           "reference_reply": ["Try logging out.", "Please DM us."],
                           "intent": [auto_ok, intents.OTHER], "should_escalate": [False, True],
                           "escalation_reason": ["none", "safety_legal"], "notes": ["", ""]})
    gp = tmp_path / "golden.csv"; golden.to_csv(gp, index=False)
    corpus = pd.DataFrame({"pair_id": [0], "customer_text": ["playlist disappeared"], "brand_reply": ["Try logging out."]})
    cp = tmp_path / "corpus.csv"; corpus.to_csv(cp, index=False)

    def transport(model, messages, temperature, json_mode, max_tokens):
        if "quality reviewer" in messages[0]["content"]:
            return json.dumps({"groundedness": 4, "correctness": 4, "tone": 4, "actionability": 4, "overall": 4, "rationale": "ok"})
        return json.dumps({"intent": auto_ok, "confidence": 0.9, "draft_reply": "Try logging out.", "escalate": False, "reason": "std"})

    llm = LLMClient(tmp_path / "c.jsonl", offline=False, transport=transport, rpm=100000)
    m = run_eval(gp, tmp_path / "out", corpus_path=cp, llm=llm)
    assert (tmp_path / "out" / "metrics.json").exists() and (tmp_path / "out" / "summary.md").exists()
    assert m["intent"]["agent"]["accuracy"] == 0.5 and m["escalation"]["agent"]["recall"] == 1.0
    preds = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert set(preds.columns) >= {"golden_id", "agent_intent", "agent_escalate", "agent_reply", "nn_reply", "canned_reply"}


def test_partial_run_cannot_clobber_committed_results(tmp_path):
    # --limit and --skip-judge both write a results dir that looks complete but is not.
    # Pointed at the committed results/, that silently replaces the headline numbers.
    import pytest

    from support_agent import config
    from support_agent.eval.run import _guard_partial_overwrite

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        _guard_partial_overwrite(config.RESULTS_DIR, 20, False, force=False)
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        _guard_partial_overwrite(config.RESULTS_DIR, None, True, force=False)

    # A different --out is always fine, a full run is always fine, and --force is the escape hatch.
    _guard_partial_overwrite(tmp_path / "results_smoke", 20, True, force=False)
    _guard_partial_overwrite(config.RESULTS_DIR, None, False, force=False)
    _guard_partial_overwrite(config.RESULTS_DIR, 20, True, force=True)


def test_stats_block_accompanies_every_headline_metric(tmp_path, monkeypatch):
    # Every point estimate the report quotes must ship with an interval next to it.
    import numpy as np
    import pandas as pd

    from support_agent.eval.run import _stats

    from support_agent import intents

    rng = np.random.default_rng(0)
    n = 40
    x, y = intents.INTENT_NAMES[0], intents.INTENT_NAMES[1]
    golden = pd.DataFrame({"golden_id": range(n), "intent": [x, y] * (n // 2),
                           "should_escalate": [True, False] * (n // 2)})
    preds = pd.DataFrame({"golden_id": range(n), "majority_intent": [x] * n,
                          "logreg_intent": list(rng.choice([x, y], n)),
                          "agent_intent": [x, y] * (n // 2),
                          "always_escalate": [True] * n, "rules_escalate": [False] * n,
                          "agent_escalate": [True, False] * (n // 2)})
    judged = pd.DataFrame({"golden_id": list(range(n)) * 2, "system": ["agent"] * n + ["nn"] * n,
                           "overall": list(rng.integers(1, 6, n)) + list(rng.integers(1, 6, n))})
    s = _stats(golden, preds, judged)
    for sysname in ("majority", "logreg", "agent"):
        assert len(s["intent"][sysname]["accuracy_ci95"]) == 2
        assert len(s["intent"][sysname]["macro_f1_ci95"]) == 2
    for sysname in ("always", "rules_only", "agent"):
        assert len(s["escalation"][sysname]["weighted_error_ci95"]) == 2
    assert s["intent"]["agent_vs_logreg_mcnemar"]["p_exact"] <= 1.0
    assert "ci95" in s["escalation"]["agent_vs_always"]
    assert "ci95" in s["reply"]["agent_vs_nn"]


def test_stats_block_survives_a_judgeless_run(tmp_path):
    # --skip-judge produces an empty judged frame; the stats pass must not blow up on it.
    import pandas as pd

    from support_agent.eval.run import _stats

    from support_agent import intents

    x, y = intents.INTENT_NAMES[0], intents.INTENT_NAMES[1]
    golden = pd.DataFrame({"golden_id": [0, 1], "intent": [x, y], "should_escalate": [True, False]})
    preds = pd.DataFrame({"golden_id": [0, 1], "majority_intent": [x, x], "logreg_intent": [x, y],
                          "agent_intent": [x, y], "always_escalate": [True, True],
                          "rules_escalate": [False, False], "agent_escalate": [True, False]})
    s = _stats(golden, preds, pd.DataFrame())
    assert s["reply"] == {}
    assert s["intent"]["agent"]["accuracy_ci95"]
