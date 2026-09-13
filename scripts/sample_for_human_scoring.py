"""Step 1 of the judge-agreement study: draw a blind human-scoring sheet.

Picks 60 (golden_id, system) rows from `results/predictions.csv` -- 30 agent, 20 nearest-neighbour,
10 canned -- shuffles them, and writes `data/golden/human_scoring_sheet.csv` with the customer
message, the brand's reference reply and the candidate reply, and *nothing else*. The sheet
deliberately carries no judge score, no rationale and no system-revealing text beyond the `system`
column, which is kept only so `scripts/judge_agreement.py` can join back on (golden_id, system);
the scorer is instructed not to read it (see `docs/05-judge-agreement.md`).

Each golden_id is used at most once. Sampling the three systems independently would have shown the
same customer message two or three times with different candidates, which turns an absolute
rubric score into a comparative one -- the scorer would rank the candidates against each other
instead of against the rubric. Disjoint ids cost nothing here (60 <= 160) and keep every row an
independent judgement.

One eligibility constraint: `results/judge_scores.csv` covers agent and nn on all 160 golden rows
but canned on golden_ids 0-39 only (the canned reply is a single constant string, so it was judged
on a 40-row subsample to save budget). A human label with no judge score on the other side of the
join is worthless, so the canned slice is restricted to pairs the judge actually scored. This
script reads *only* the (golden_id, system) key columns of the judge file -- `usecols` makes that
mechanical -- so no score, rationale or candidate from the judge run can reach the sheet.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from support_agent import config

# 30/20/10: the agent is the system under test and gets the tightest error bar; nn is the
# retrieval baseline it must beat; canned is a near-constant control that mainly checks the
# scorer and the judge both punish a generic reply. See decision log entry 9.
QUOTA = {"agent": 30, "nn": 20, "canned": 10}
REPLY_COLUMN = {"agent": "agent_reply", "nn": "nn_reply", "canned": "canned_reply"}


def judged_pairs(judge_path) -> dict[str, set[int]]:
    """golden_ids the judge scored, per system. Reads the key columns only -- never a score."""
    keys = pd.read_csv(judge_path, usecols=["golden_id", "system"])
    return {s: set(g.golden_id) for s, g in keys.groupby("system")}


def build_sheet(preds: pd.DataFrame, golden: pd.DataFrame, eligible: dict[str, set[int]],
                seed: int = config.SEED) -> pd.DataFrame:
    merged = preds.merge(golden[["golden_id", "reference_reply"]], on="golden_id", how="inner")
    total = sum(QUOTA.values())
    if len(merged) < total:
        raise ValueError(f"need at least {total} prediction rows, have {len(merged)}")

    rng = np.random.default_rng(seed)
    ids = merged["golden_id"].tolist()
    picked = list(rng.choice(ids, size=total, replace=False))

    # Slice the pool in order, then drop any id the system in question was not judged on and top the
    # slice back up from that system's remaining eligible ids (same seeded stream, so still
    # reproducible). Only the canned slice is ever short; agent and nn are judged on every row.
    assignment: dict[str, list[int]] = {}
    used: set[int] = set()
    cursor = 0
    for system, quota in QUOTA.items():
        pool = [i for i in picked[cursor:cursor + quota] if i in eligible.get(system, set(ids))]
        cursor += quota
        used.update(pool)
        short = quota - len(pool)
        if short:
            spare = sorted(eligible.get(system, set(ids)) - used - set(picked))
            if len(spare) < short:
                raise ValueError(f"{system}: need {quota} judged rows, only {len(pool) + len(spare)} available")
            extra = list(rng.choice(spare, size=short, replace=False))
            pool += extra
            used.update(extra)
        assignment[system] = pool

    rows = []
    for system in QUOTA:
        for golden_id in assignment[system]:
            r = merged.loc[merged.golden_id == golden_id].iloc[0]
            candidate = r[REPLY_COLUMN[system]]
            rows.append({
                "golden_id": golden_id,
                "system": system,
                "customer_text": r.customer_text,
                "reference_reply": r.reference_reply,
                "candidate": "" if pd.isna(candidate) else str(candidate),
            })

    sheet = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet.insert(0, "sheet_id", [f"H{i:03d}" for i in range(1, len(sheet) + 1)])
    return sheet


def main() -> None:
    pred_path = config.RESULTS_DIR / "predictions.csv"
    if not pred_path.exists():
        print(f"{pred_path} not found; run `python -m support_agent.eval.run` first.", file=sys.stderr)
        sys.exit(1)

    eligible = judged_pairs(config.RESULTS_DIR / "judge_scores.csv")
    sheet = build_sheet(pd.read_csv(pred_path), pd.read_csv(config.GOLDEN_DIR / "golden_set.csv"), eligible)
    out = config.GOLDEN_DIR / "human_scoring_sheet.csv"
    sheet.to_csv(out, index=False)
    print(f"wrote {out} ({len(sheet)} rows)")
    print(sheet.system.value_counts().to_string())


if __name__ == "__main__":
    main()
