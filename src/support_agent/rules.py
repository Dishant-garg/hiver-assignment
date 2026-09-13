"""Deterministic escalation rules. They exist because an LLM's judgement about *its own*
confidence is not evidence; these rules are the part of the decision a human can audit.

pre_check runs on the raw message before any model call (hard triggers).
post_check runs on the model output (soft triggers: low confidence, weak evidence, unsafe draft).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from support_agent import intents

ESCALATION_CATEGORIES = ["none", "needs_account_access", "billing_dispute", "safety_legal",
                         "abusive", "ambiguous", "out_of_scope", "human_requested", "pii"]


@dataclass(frozen=True)
class RuleHit:
    rule: str
    reason: str


# These four implement section (b) rules 1-4 of `data/golden/labeling-guide.md`, which was fixed
# before labelling started. Where a pattern below looks oddly specific, it is tracking the guide's
# own wording -- in particular its two explicit carve-outs, marked GUIDE b2 and GUIDE b4.
_PRE = [
    ("human_requested", "Customer explicitly asked for a human",
     # GUIDE b4: "asks for a real person, an agent, a supervisor, OR TO BE PHONED/DM'D by a
     # human". The DM half was missing, which mattered: a bare "please see my DM" is the single
     # most common way this brand's customers ask for a person, and the rule saw none of them.
     re.compile(r"\b(real|actual|human|live)\s+(person|agent|human|rep|representative)\b"
                r"|\b(speak|talk|chat)\s+(to|with)\s+(someone|somebody|an?\s+(real\s+)?(agent|human|person|rep))\b"
                r"|\banyone\s+(live|real)\b|\bsupervisor\b|\bmanager\b"
                r"|\b(dm|pm|inbox|message)\s+(me|us)\b"
                r"|\b(please|pls|plz|can\s+(you|we)|could\s+you|will\s+you)\b[^.!?]{0,30}\b(dm|pm|direct\s+message)\b"
                r"|\b(see|answer|check|reply\s+to|respond\s+to)\s+(my|the)\s+(dm|dms|direct\s+message)\b"
                r"|\b(talk|speak)\s+in\s+(dm|dms)\b"
                r"|\b(call|phone|ring)\s+me\b", re.I)),
    ("safety_legal", "Legal, fraud, or safety language",
     # GUIDE b1 also names phishing-style fraud and discrimination, neither of which the original
     # pattern covered.
     re.compile(r"\b(sue|suing|lawsuit|lawyer|attorney|legal action|fraud\w*|scam|scammed|phish\w*|spoof\w*"
                r"|police|harass\w*|discriminat\w*|sexis\w*|racis\w*|homophob\w*|misogyn\w*"
                r"|unsafe|injur\w*|threat\w*|suicid\w*|self[- ]harm)\b", re.I)),
    ("pii", "Message contains a card number, email address, or phone number",
     re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b|\b\d{15,16}\b|[\w.+-]+@[\w-]+\.[\w.]+|(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]\d{3}[ -]\d{4}\b")),
    ("abusive", "Abusive language",
     # GUIDE b2: "Frustrated swearing *about the product* ('this damn app') is not abuse on its
     # own; abuse aimed at a person is." The original pattern matched bare profanity, so it called
     # "your web player fucking blows" abuse -- the guide's own counter-example. Profanity now
     # only counts when it is pointed at a person, and the insult branch is unchanged.
     re.compile(r"\b(fuck|screw)\s+(you|u|off|yourself|urself)\b"
                r"|\byou\s+(fucking\s+)?(idiots?|morons?|assholes?|pricks?|clowns?)\b"
                r"|\b(asshole|bastard|prick|moron)s?\b"
                r"|\b(you|you're|youre|ur|you guys|yall|y'all|your (team|staff|company))\s+(are\s+)?(all\s+)?((f\*+ing|fucking|bloody|damn)\s+)?(useless|idiots?|garbage|trash|pathetic|incompetent)\b", re.I)),
]

_PROMISE = re.compile(r"\b(full refund|refund(ed)? (of )?\$?\d|guarantee[ds]?|will (be )?(credit|reimburse)\w*|compensat\w+)\b|\b(refund|credit|reimburse\w*|money)\b[^.!?]{0,40}\b(within \d+ (hours|days)|by (mon|tues|wednes|thurs|fri|satur|sun)day)\b", re.I)


# The pre_check rule names, in order. Anything that has to reconstruct a decision from a
# recorded `rule_hits` list (eval.tune) needs this instead of its own literal copy.
PRE_RULE_NAMES = tuple(name for name, _, _ in _PRE)


def pre_check(message: str) -> list[RuleHit]:
    return [RuleHit(name, reason) for name, reason, rx in _PRE if rx.search(message)]


def post_check(intent_name: str, confidence: float, retrieval_score: float, draft: str, *,
               conf_threshold: float, retr_threshold: float, max_chars: int) -> list[RuleHit]:
    out: list[RuleHit] = []
    if confidence < conf_threshold:
        out.append(RuleHit("low_confidence", f"Classifier confidence {confidence:.2f} < {conf_threshold}"))
    if retrieval_score < retr_threshold:
        out.append(RuleHit("weak_evidence", f"Best historical match {retrieval_score:.2f} < {retr_threshold}"))
    if not intents.by_name(intent_name).auto_handle_allowed:
        out.append(RuleHit("intent_not_auto_handleable", f"Intent '{intent_name}' requires a human (account, money, or policy)"))
    if _PROMISE.search(draft):
        out.append(RuleHit("forbidden_promise", "Draft promises money, timing, or a guarantee the brand never makes"))
    if len(draft) > max_chars:
        out.append(RuleHit("too_long", f"Draft is {len(draft)} chars > {max_chars}"))
    return out
