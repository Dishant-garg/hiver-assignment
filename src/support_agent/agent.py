"""The support agent: one structured LLM call sandwiched between deterministic rules.

Why one call: the Groq free tier budget (~200K tokens/day) makes three calls per message
unaffordable across a 200-example evaluation, and intent + draft + escalation share context.
Why rules around it: the escalation decision must be auditable; the LLM's own opinion about
whether to escalate is one input, never the final word."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from support_agent import config, intents, rules
from support_agent.llm import LLMClient
from support_agent.retrieve import Evidence, Retriever

SYSTEM_PROMPT = """You are the {brand} customer support agent on Twitter. You answer one incoming customer message.
Use ONLY the historical exchanges provided as evidence for facts, steps, and policy. Never promise refunds, credits, dates, or guarantees. Never ask for passwords or card numbers. Match the brand's friendly, concise Twitter tone. Replies must be under 280 characters.

Intents (choose exactly one):
{intent_block}

Respond with a single JSON object with keys:
"intent" (one of the intent names), "confidence" (0.0-1.0, how sure you are of the intent),
"draft_reply" (the reply text), "escalate" (true if a human must handle this: account or billing access needed, legal/safety, anger, unclear request, or no relevant evidence),
"reason" (one sentence explaining the escalate decision)."""

USER_TEMPLATE = """Historical exchanges (evidence):
{evidence}

Incoming customer message:
"{message}"

Return the JSON object."""


def _format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none)"
    return "\n".join(f'{i + 1}. customer: "{e.customer_text[:280]}"\n   {config.BRAND}: "{e.brand_reply[:280]}"'
                     for i, e in enumerate(evidence))


def build_messages(message: str, evidence: list[Evidence]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(brand=config.BRAND, intent_block=intents.definitions_block())},
        {"role": "user", "content": USER_TEMPLATE.format(evidence=_format_evidence(evidence), message=message)},
    ]


def parse_llm_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return default


@dataclass
class AgentDecision:
    message: str
    intent: str
    confidence: float
    draft_reply: str
    escalate: bool
    reason: str
    evidence: list[Evidence] = field(default_factory=list)
    rule_hits: list[str] = field(default_factory=list)
    llm_escalate: bool = False
    llm_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d


class SupportAgent:
    def __init__(self, llm: LLMClient, retriever: Retriever, *, model: str = config.AGENT_MODEL,
                 top_k: int = config.TOP_K, conf_threshold: float = config.CONFIDENCE_THRESHOLD,
                 retr_threshold: float = config.RETRIEVAL_THRESHOLD, max_chars: int = config.MAX_REPLY_CHARS):
        self.llm, self.retriever, self.model, self.top_k = llm, retriever, model, top_k
        self.conf_threshold, self.retr_threshold, self.max_chars = conf_threshold, retr_threshold, max_chars

    def handle(self, message: str) -> AgentDecision:
        pre = rules.pre_check(message)
        evidence = self.retriever.top_k(message, self.top_k)
        raw = parse_llm_json(self.llm.chat(self.model, build_messages(message, evidence), tag="agent"))

        intent = raw.get("intent") if raw.get("intent") in intents.INTENT_NAMES else intents.OTHER
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        draft = str(raw.get("draft_reply") or "").strip()
        llm_escalate = _as_bool(raw.get("escalate", True), True)
        llm_reason = str(raw.get("reason") or "").strip()
        if not raw:
            confidence, llm_escalate, llm_reason = 0.0, True, "Model returned unparseable output"

        top_score = evidence[0].score if evidence else 0.0
        post = rules.post_check(intent, confidence, top_score, draft, conf_threshold=self.conf_threshold,
                                retr_threshold=self.retr_threshold, max_chars=self.max_chars)
        hits = pre + post
        escalate = bool(hits) or llm_escalate
        reasons = [h.reason for h in hits] + ([llm_reason] if llm_escalate and llm_reason else [])
        if escalate:
            reason = "; ".join(reasons) or "Model requested escalation without a stated reason"
        else:
            reason = llm_reason or "Confident, grounded, auto-handleable intent"
        return AgentDecision(message, intent, confidence, draft, escalate, reason, evidence,
                             [h.rule for h in hits], llm_escalate, llm_reason)
