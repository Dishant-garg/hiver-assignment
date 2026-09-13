"""Grid-search the two post-check thresholds on the dev slice (never the golden set).
The agent LLM call is made once per dev message; thresholds are re-applied post-hoc."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

from support_agent import config, rules
from support_agent.agent import SupportAgent
from support_agent.eval.metrics import escalation_metrics
from support_agent.llm import DailyLimitError, LLMClient
from support_agent.retrieve import Retriever


def reapply(decision, conf_threshold: float, retr_threshold: float) -> bool:
    """Re-derive `decision.escalate` at a different threshold pair, without a second LLM call.

    Mirrors `SupportAgent.handle`: pre-rule hits survive (they never depended on a threshold),
    post-rules are recomputed, and the model's own escalate flag still overrides both.
    """
    post = rules.post_check(decision.intent, decision.confidence,
                            decision.evidence[0].score if decision.evidence else 0.0, decision.draft_reply,
                            conf_threshold=conf_threshold, retr_threshold=retr_threshold,
                            max_chars=config.MAX_REPLY_CHARS)
    pre = [h for h in decision.rule_hits if h in rules.PRE_RULE_NAMES]
    return bool(pre or post) or decision.llm_escalate


def tune_thresholds(dev_path: Path, llm: LLMClient | None = None) -> dict:
    dev = pd.read_csv(dev_path)
    agent = SupportAgent(llm or LLMClient(), Retriever.from_csv(config.brand_dir() / "corpus.csv"),
                         conf_threshold=0.0, retr_threshold=0.0)
    decisions = []
    for t in dev.customer_text:
        try:
            decisions.append(agent.handle(t))
        except DailyLimitError as e:
            print(f"\nDaily cap hit after {len(decisions)} rows: {e}\nRe-run later; cached rows are kept.", file=sys.stderr)
            sys.exit(1)
    y = [bool(v) for v in dev.should_escalate]
    best = None
    for c, r in itertools.product([0.0, 0.5, 0.6, 0.7, 0.8, 0.9], [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]):
        preds = [reapply(d, c, r) for d in decisions]
        m = escalation_metrics(y, preds)
        cand = {"conf_threshold": c, "retr_threshold": r, **{k: m[k] for k in ["weighted_error", "recall", "automation_rate"]}}
        if best is None or (cand["weighted_error"], -cand["automation_rate"]) < (best["weighted_error"], -best["automation_rate"]):
            best = cand
    return best


if __name__ == "__main__":
    best = tune_thresholds(config.GOLDEN_DIR / "dev_set.csv")
    print(json.dumps(best, indent=2))
    print("Set CONFIDENCE_THRESHOLD / RETRIEVAL_THRESHOLD defaults in config.py to these values.", file=sys.stderr)
