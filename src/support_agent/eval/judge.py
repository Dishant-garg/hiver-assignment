"""LLM-as-judge for reply quality. The judge sees the message, the brand's real reply
(reference), the evidence the candidate was allowed to use, and the candidate -- never which
system produced it. Scores are 1-5 ints; 'overall' is the judge's holistic score and is what we
compare against human labels."""
from __future__ import annotations

import json
from dataclasses import dataclass

from support_agent import config
from support_agent.agent import parse_llm_json
from support_agent.llm import LLMClient

RUBRIC = """Score the candidate reply from 1 (bad) to 5 (excellent) on each criterion:
- groundedness: uses only facts/steps that appear in the evidence or reference; 1 = invents policy, steps, or promises.
- correctness: would resolve or correctly advance the customer's issue, judged against the reference reply; 1 = wrong or irrelevant.
- tone: friendly, concise, on-brand for Twitter support, no blame; 1 = rude, robotic, or rambling.
- actionability: gives the customer a concrete next step; 1 = says nothing useful.
- overall: your holistic judgement of whether this reply could be sent as-is.
Return JSON: {"groundedness": int, "correctness": int, "tone": int, "actionability": int, "overall": int, "rationale": "one sentence"}"""


@dataclass
class JudgeScore:
    groundedness: int
    correctness: int
    tone: int
    actionability: int
    overall: int
    rationale: str
    parse_ok: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_judge_messages(message: str, reference_reply: str, evidence_replies: list[str], candidate_reply: str) -> list[dict]:
    ev = "\n".join(f"- {e[:280]}" for e in evidence_replies) or "- (none)"
    user = (f'Customer message: "{message}"\n\nReference reply the brand actually sent: "{reference_reply}"\n\n'
            f'Evidence available to the candidate:\n{ev}\n\nCandidate reply: "{candidate_reply}"\n\n{RUBRIC}')
    return [{"role": "system", "content": f"You are a strict quality reviewer for {config.BRAND} customer support replies."},
            {"role": "user", "content": user}]


def _clamp(v) -> int:
    try:
        return max(1, min(5, int(round(float(v)))))
    except (TypeError, ValueError):
        return 1


def _is_number(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v)
            return True
        except ValueError:
            return False
    return False


class ReplyJudge:
    def __init__(self, llm: LLMClient, model: str = config.JUDGE_MODEL):
        self.llm, self.model = llm, model

    def score(self, message: str, reference_reply: str, evidence_replies: list[str], candidate_reply: str, *,
              temperature: float = 0.0) -> JudgeScore:
        raw = parse_llm_json(self.llm.chat(self.model, build_judge_messages(message, reference_reply, evidence_replies, candidate_reply),
                                           temperature=temperature, max_tokens=200, tag="judge"))
        score_fields = ("groundedness", "correctness", "tone", "actionability", "overall")
        parse_ok = bool(raw) and all(_is_number(raw.get(f)) for f in score_fields)
        return JudgeScore(_clamp(raw.get("groundedness")), _clamp(raw.get("correctness")), _clamp(raw.get("tone")),
                          _clamp(raw.get("actionability")), _clamp(raw.get("overall")), str(raw.get("rationale", ""))[:300],
                          parse_ok=parse_ok)
