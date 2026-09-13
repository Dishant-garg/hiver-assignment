import json

from support_agent.eval.judge import JudgeScore, ReplyJudge, build_judge_messages
from support_agent.llm import LLMClient


def test_judge_parses_scores_and_clamps(tmp_path):
    def transport(model, messages, temperature, json_mode, max_tokens):
        return json.dumps({"groundedness": 5, "correctness": 4, "tone": 9, "actionability": 0, "overall": 4, "rationale": "fine"})
    j = ReplyJudge(LLMClient(tmp_path / "c.jsonl", offline=False, transport=transport, rpm=100000), model="m")
    s = j.score("msg", "ref", ["ev1"], "cand")
    assert isinstance(s, JudgeScore) and s.tone == 5 and s.actionability == 1 and s.overall == 4
    assert s.parse_ok is True


def test_judge_flags_parse_failure(tmp_path):
    def bad_transport(model, messages, temperature, json_mode, max_tokens):
        return "not json"
    j = ReplyJudge(LLMClient(tmp_path / "c1.jsonl", offline=False, transport=bad_transport, rpm=100000), model="m")
    s = j.score("msg", "ref", ["ev1"], "cand")
    assert s.parse_ok is False
    assert (s.groundedness, s.correctness, s.tone, s.actionability, s.overall) == (1, 1, 1, 1, 1)

    def partial_transport(model, messages, temperature, json_mode, max_tokens):
        return json.dumps({"groundedness": 4, "correctness": 4, "tone": "n/a", "actionability": 3, "overall": 4, "rationale": "ok"})
    j2 = ReplyJudge(LLMClient(tmp_path / "c2.jsonl", offline=False, transport=partial_transport, rpm=100000), model="m")
    s2 = j2.score("msg", "ref", ["ev1"], "cand")
    assert s2.parse_ok is False


def test_judge_prompt_is_blind_to_system_identity():
    msgs = build_judge_messages("msg", "ref", ["ev"], "cand")
    text = json.dumps(msgs).lower()
    for leak in ("baseline", "llm", "gpt", "nearest", "canned", "system a", "system b"):
        assert leak not in text
