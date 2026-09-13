"""`eval.failures` has to survive the two partial results directories the harness can write:
`--skip-judge` (no judge_scores.csv) and `--limit` (only some golden rows)."""
import pandas as pd

from support_agent import config
from support_agent.eval import failures

PRED_COLS = ["golden_id", "customer_text", "agent_intent", "agent_confidence", "agent_reply", "agent_escalate",
             "agent_reason", "agent_rule_hits", "agent_llm_escalate", "agent_top_score", "agent_evidence"]


def _results_dir(tmp_path, *, judge: bool):
    golden = pd.read_csv(config.GOLDEN_DIR / "golden_set.csv").head(2)
    ids = list(golden.golden_id)
    rd = tmp_path / "results"
    rd.mkdir()
    rows = []
    for i, gid in enumerate(ids):
        rows.append({"golden_id": gid, "customer_text": "text", "agent_intent": "app_bug",
                     "agent_confidence": 0.9, "agent_reply": "Try a reinstall.", "agent_escalate": bool(i),
                     "agent_reason": "reason", "agent_rule_hits": "weak_evidence" if i else "",
                     "agent_llm_escalate": False, "agent_top_score": 0.1, "agent_evidence": "evidence"})
    pd.DataFrame(rows, columns=PRED_COLS).to_csv(rd / "predictions.csv", index=False)
    if judge:
        pd.DataFrame([{"golden_id": gid, "system": "agent", "candidate": "x", "groundedness": 2,
                       "correctness": 2, "tone": 3, "actionability": 1, "overall": 2,
                       "rationale": "thin", "parse_ok": True} for gid in ids]
                     ).to_csv(rd / "judge_scores.csv", index=False)
    return rd


def test_limit_style_results_dir_with_judge_scores(tmp_path):
    md = failures.build(failures.load(_results_dir(tmp_path, judge=True)))
    assert "# Agent failure cases" in md and "n = 2." in md
    assert "## D. Lowest-judged agent replies (2 scored overall" in md
    assert "- Judge: overall 2," in md


def test_skip_judge_results_dir_degrades_instead_of_raising(tmp_path, capsys):
    df = failures.load(_results_dir(tmp_path, judge=False))
    assert "overall" not in df.columns
    md = failures.build(df)
    assert "## D. Lowest-judged agent replies (unavailable)" in md
    assert "- Judge: overall" not in md
    err = capsys.readouterr().err
    assert "no judge_scores.csv" in err and "skipping section D" in err
