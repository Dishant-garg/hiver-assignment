"""Escalation baselines. Trivial: escalate everything (100% recall, zero automation).
Simple: the deterministic rules layer alone, no model."""
from support_agent import rules


class AlwaysEscalate:
    def decide(self, message: str) -> tuple[bool, str]:
        return True, "Policy: escalate everything"


class RulesOnlyEscalation:
    def decide(self, message: str) -> tuple[bool, str]:
        hits = rules.pre_check(message)
        return (True, "; ".join(h.reason for h in hits)) if hits else (False, "No rule triggered")
