"""The grid search re-derives each decision post hoc instead of calling the LLM again.

That is only legitimate if the re-derivation agrees with a real run at the same thresholds,
so this test pins `tune.reapply` against `SupportAgent.handle` at the shipped thresholds.
"""
import json

import pandas as pd

from support_agent import config, rules
from support_agent.agent import SupportAgent
from support_agent.eval import tune
from support_agent.llm import LLMClient
from support_agent.retrieve import Retriever

CORPUS = pd.DataFrame({
    "pair_id": [0, 1],
    "customer_text": ["my playlist disappeared after the update", "i was charged twice this month"],
    "brand_reply": ["Try logging out and back in.", "Please DM us and we will take a look."],
})

# (message, should_escalate) — one clean auto-handle, one pre-rule hit, one row with no
# retrieval support at all, which is the row the retrieval threshold is supposed to catch.
DEV_ROWS = [
    ("my playlist disappeared after the update", False),
    ("my playlist disappeared and I want to talk to a real person", True),
    ("the bluetooth speaker in my car keeps stuttering", True),
]

REPLY = {"intent": "playback_issue", "confidence": 0.9, "draft_reply": "Try a reinstall and let us know.",
         "escalate": False, "reason": "standard fix"}


def _transport(model, messages, temperature, json_mode, max_tokens):
    return json.dumps(REPLY)


def _agent(tmp_path, conf, retr):
    llm = LLMClient(tmp_path / "cache.jsonl", offline=False, transport=_transport, rpm=1000000)
    return SupportAgent(llm, Retriever(CORPUS), conf_threshold=conf, retr_threshold=retr)


def test_pre_rule_names_match_the_pre_check_rules():
    # "fuck you" rather than "this is shit": guide (b2) says swearing *about the product* is not
    # abuse, so a message that only swears no longer fires the rule and would not cover `abusive`.
    hit_names = {h.rule for m in ["I want a real person", "I will sue you", "a@b.com", "fuck you"]
                 for h in rules.pre_check(m)}
    assert set(rules.PRE_RULE_NAMES) == {"human_requested", "safety_legal", "pii", "abusive"} == hit_names


def test_reapply_matches_a_real_run_at_the_shipped_thresholds(tmp_path):
    dev = tmp_path / "dev.csv"
    pd.DataFrame(DEV_ROWS, columns=["customer_text", "should_escalate"]).to_csv(dev, index=False)
    rows = pd.read_csv(dev)

    # How tune sees the world: one pass with both gates open, decisions re-scored afterwards.
    open_agent = _agent(tmp_path, 0.0, 0.0)
    shipped_agent = _agent(tmp_path, config.CONFIDENCE_THRESHOLD, config.RETRIEVAL_THRESHOLD)

    reapplied, real = [], []
    for text in rows.customer_text:
        reapplied.append(tune.reapply(open_agent.handle(text),
                                      config.CONFIDENCE_THRESHOLD, config.RETRIEVAL_THRESHOLD))
        real.append(shipped_agent.handle(text).escalate)

    assert reapplied == real
    # The test is only worth anything if the thresholds actually changed a decision.
    assert real != [tune.reapply(open_agent.handle(t), 0.0, 0.0) for t in rows.customer_text]
