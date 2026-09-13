import json

import pandas as pd

from support_agent.agent import AgentDecision, SupportAgent, parse_llm_json
from support_agent.llm import LLMClient
from support_agent.retrieve import Retriever
from support_agent import intents


def make_agent(tmp_path, reply):
    def transport(model, messages, temperature, json_mode, max_tokens):
        return json.dumps(reply)
    llm = LLMClient(tmp_path / "c.jsonl", offline=False, transport=transport, rpm=100000)
    corpus = pd.DataFrame({"pair_id": [0], "customer_text": ["my playlist disappeared after the update"],
                           "brand_reply": ["Try logging out and back in."]})
    return SupportAgent(llm, Retriever(corpus), conf_threshold=0.6, retr_threshold=0.1)


def test_auto_handle_when_confident_and_grounded(tmp_path):
    auto_ok = next(i.name for i in intents.INTENTS if i.auto_handle_allowed)
    a = make_agent(tmp_path, {"intent": auto_ok, "confidence": 0.9, "draft_reply": "Try logging out and back in.",
                              "escalate": False, "reason": "standard fix"})
    d = a.handle("my playlist disappeared after the update")
    assert isinstance(d, AgentDecision) and d.escalate is False and d.intent == auto_ok
    assert d.rule_hits == [] and "standard fix" in d.reason


def test_pre_rule_forces_escalation_even_if_llm_says_no(tmp_path):
    auto_ok = next(i.name for i in intents.INTENTS if i.auto_handle_allowed)
    a = make_agent(tmp_path, {"intent": auto_ok, "confidence": 0.95, "draft_reply": "ok", "escalate": False, "reason": "x"})
    d = a.handle("my playlist disappeared, I want to talk to a real person")
    assert d.escalate is True and "human_requested" in d.rule_hits and "human" in d.reason.lower()


def test_low_confidence_escalates_and_unknown_intent_maps_to_other(tmp_path):
    a = make_agent(tmp_path, {"intent": "made_up", "confidence": 0.2, "draft_reply": "ok", "escalate": False, "reason": "x"})
    d = a.handle("my playlist disappeared after the update")
    assert d.intent == intents.OTHER and d.escalate is True and "low_confidence" in d.rule_hits


def test_parse_llm_json_tolerates_fences_and_garbage():
    assert parse_llm_json('```json\n{"a": 1}\n```')["a"] == 1
    assert parse_llm_json("not json") == {}
    assert parse_llm_json("[1, 2]") == {}
    assert parse_llm_json("null") == {}


def test_llm_escalation_with_blank_reason_still_states_a_reason(tmp_path):
    auto_ok = next(i.name for i in intents.INTENTS if i.auto_handle_allowed)
    a = make_agent(tmp_path, {"intent": auto_ok, "confidence": 0.9, "draft_reply": "ok",
                              "escalate": True, "reason": ""})
    d = a.handle("my playlist disappeared after the update")
    assert d.escalate is True and d.reason != ""


def test_string_escalate_false_is_treated_as_false(tmp_path):
    auto_ok = next(i.name for i in intents.INTENTS if i.auto_handle_allowed)
    a = make_agent(tmp_path, {"intent": auto_ok, "confidence": 0.9, "draft_reply": "Try logging out and back in.",
                              "escalate": "false", "reason": "standard fix"})
    d = a.handle("my playlist disappeared after the update")
    assert d.escalate is False
