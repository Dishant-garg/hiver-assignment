"""Judge self-consistency check (Step 6.4): re-scores the agent reply for the first 40 golden
rows at temperature 0.0 (as used everywhere else) and 0.7, and reports how much the judge's
'overall' score moves when nothing about the input changed except sampling temperature. High
agreement here is what lets us trust a single temperature-0.0 judge pass elsewhere."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from support_agent import config
from support_agent.eval.agreement import agreement_report
from support_agent.eval.judge import ReplyJudge
from support_agent.llm import DailyLimitError, LLMClient

CONSTANT_NOTE = ("kappa_quadratic is undefined (reported as 0.0, the conservative reading) when "
                 "overall_t0 or overall_t07 has a single distinct value -- a likely outcome for a "
                 "self-consistent judge on a small, easy sample. Read exact_agreement and within_one instead.")


def main() -> None:
    pred_path = config.RESULTS_DIR / "predictions.csv"
    if not pred_path.exists():
        print(f"{pred_path} not found; run `python -m support_agent.eval.run` first.", file=sys.stderr)
        sys.exit(1)
    golden = pd.read_csv(config.GOLDEN_DIR / "golden_set.csv")
    preds = pd.read_csv(pred_path)
    merged = preds.merge(golden[["golden_id", "reference_reply"]], on="golden_id").head(40)

    llm = LLMClient()
    judge = ReplyJudge(llm)
    rows = []
    for _, r in merged.iterrows():
        ev = [] if pd.isna(r.agent_evidence) else [e for e in str(r.agent_evidence).split(" || ") if e]
        reply = "" if pd.isna(r.agent_reply) else str(r.agent_reply)
        try:
            s0 = judge.score(r.customer_text, r.reference_reply, ev, reply, temperature=0.0)
            s07 = judge.score(r.customer_text, r.reference_reply, ev, reply, temperature=0.7)
        except DailyLimitError as e:
            print(f"\nDaily cap hit after {len(rows)} rows: {e}\nRe-run later; cached rows are kept.", file=sys.stderr)
            break
        rows.append({"golden_id": r.golden_id, "overall_t0": s0.overall, "overall_t07": s07.overall})
        print(f"\r  judge_consistency {len(rows)}/{len(merged)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    if not rows:
        print("Daily cap hit before any row was scored; nothing written. Re-run later.", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.RESULTS_DIR / "judge_consistency.csv", index=False)

    report = agreement_report(list(out.overall_t0), list(out.overall_t07))
    constant_t0 = out.overall_t0.nunique() <= 1
    constant_t07 = out.overall_t07.nunique() <= 1
    report["constant_t0"] = bool(constant_t0)
    report["constant_t07"] = bool(constant_t07)
    report["note"] = CONSTANT_NOTE if (constant_t0 or constant_t07) else ""
    (config.RESULTS_DIR / "judge_consistency.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
