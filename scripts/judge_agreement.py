"""Step 3 of the judge-agreement study: how well does the LLM judge's `overall` track a human
rater's `overall` on the same 60 replies?

Joins `data/golden/human_judge_labels.csv` to `results/judge_scores.csv` on (golden_id, system),
runs `agreement.agreement_report` overall and per system, copies in the temperature self-consistency
figures from `results/judge_consistency.json`, writes `results/judge_agreement.json`, and prints a
markdown table plus the largest disagreements (which is what `docs/05-judge-agreement.md` quotes).

Every kappa and Spearman also gets a seeded percentile bootstrap 95% interval, because the point
estimates alone invite over-reading. Per-system kappa is reported because it is asked for, not
because it is reliable: the measured intervals are [0.339, 0.745] for agent, [0.137, 0.751] for nn
and [-0.182, 0.623] for canned, and on the canned slice the scores are nearly constant, which
deflates kappa mechanically (`weighted_kappa` returns 0.0 outright for a constant vector). Read the
overall row.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from support_agent import config
from support_agent.eval.agreement import agreement_report, spearman, weighted_kappa

SYSTEMS = ["agent", "nn", "canned"]
BOOTSTRAP_RESAMPLES = 2000
KAPPA_CAVEAT = ("Per-system kappa is computed on n=30/20/10 and is unstable at those sizes; the "
                "canned slice in particular has almost no score variance, which deflates kappa "
                "mechanically. Use the overall row for the headline number.")
BOOTSTRAP_NOTE = (f"kappa_ci95 and spearman_ci95 are seeded percentile bootstrap intervals "
                  f"(config.SEED, {BOOTSTRAP_RESAMPLES} resamples of the scored pairs, 2.5th/97.5th "
                  "percentile). A resample in which either score vector comes out constant has an "
                  "undefined kappa; agreement.weighted_kappa returns 0.0 there, the conservative "
                  "reading, so such resamples pull the lower bound down rather than being dropped.")


def bootstrap_ci(human: list[int], judge: list[int], seed: int = config.SEED) -> dict[str, list[float]]:
    """Percentile bootstrap over the (human, judge) pairs. A fresh generator per call keeps each
    slice's interval reproducible on its own, independent of the order slices are computed in."""
    rng = np.random.default_rng(seed)
    h, j = np.array(human), np.array(judge)
    kappas, rhos = [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, len(h), size=len(h))
        kappas.append(weighted_kappa(list(h[idx]), list(j[idx])))
        rhos.append(spearman(list(h[idx]), list(j[idx])))
    return {
        "kappa_ci95": [round(float(np.percentile(kappas, 2.5)), 3), round(float(np.percentile(kappas, 97.5)), 3)],
        "spearman_ci95": [round(float(np.percentile(rhos, 2.5)), 3), round(float(np.percentile(rhos, 97.5)), 3)],
    }


def load() -> pd.DataFrame:
    human = pd.read_csv(config.GOLDEN_DIR / "human_judge_labels.csv")
    judge = pd.read_csv(config.RESULTS_DIR / "judge_scores.csv")
    merged = human.merge(judge[["golden_id", "system", "candidate", "overall", "rationale", "parse_ok"]],
                         on=["golden_id", "system"], how="left", validate="one_to_one")
    missing = merged.overall.isna().sum()
    if missing:
        raise ValueError(f"{missing} human-labelled rows have no judge score for (golden_id, system)")
    merged = merged.rename(columns={"overall": "judge_overall"})
    merged["judge_overall"] = merged.judge_overall.astype(int)
    merged["delta"] = merged.judge_overall - merged.human_overall
    return merged


def report_for(df: pd.DataFrame) -> dict:
    rep = agreement_report(list(df.human_overall), list(df.judge_overall))
    rep["human_mean"] = round(float(df.human_overall.mean()), 3)
    rep["judge_mean"] = round(float(df.judge_overall.mean()), 3)
    rep["judge_minus_human_mean"] = round(float(df.delta.mean()), 3)
    rep.update(bootstrap_ci(list(df.human_overall), list(df.judge_overall)))
    return rep


def markdown_table(overall: dict, per_system: dict[str, dict]) -> str:
    head = ("| slice | n | kappa (quadratic) | kappa 95% CI | Spearman | Spearman 95% CI | exact | "
            "within 1 | human mean | judge mean | judge - human |")
    sep = "|---|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|"
    rows = [head, sep]
    for name, rep in [("overall", overall)] + [(s, per_system[s]) for s in SYSTEMS]:
        kl, kh = rep["kappa_ci95"]
        sl, sh = rep["spearman_ci95"]
        rows.append(f"| {name} | {rep['n']} | {rep['kappa_quadratic']:.3f} | [{kl:.3f}, {kh:.3f}] | "
                    f"{rep['spearman']:.3f} | [{sl:.3f}, {sh:.3f}] | "
                    f"{rep['exact_agreement']:.2f} | {rep['within_one']:.2f} | {rep['human_mean']:.2f} | "
                    f"{rep['judge_mean']:.2f} | {rep['judge_minus_human_mean']:+.2f} |")
    return "\n".join(rows)


def main() -> None:
    for p in [config.GOLDEN_DIR / "human_judge_labels.csv", config.RESULTS_DIR / "judge_scores.csv"]:
        if not p.exists():
            print(f"{p} not found.", file=sys.stderr)
            sys.exit(1)

    df = load()
    overall = report_for(df)
    per_system = {s: report_for(df[df.system == s]) for s in SYSTEMS}

    consistency_path = config.RESULTS_DIR / "judge_consistency.json"
    consistency = json.loads(consistency_path.read_text()) if consistency_path.exists() else {}

    out = {
        "overall": overall,
        "per_system": per_system,
        "self_consistency": consistency,
        "system_ranking": {
            "human": df.groupby("system").human_overall.mean().round(3).sort_values(ascending=False).to_dict(),
            "judge": df.groupby("system").judge_overall.mean().round(3).sort_values(ascending=False).to_dict(),
        },
        "judge_parse_ok": bool(df.parse_ok.all()),
        "kappa_caveat": KAPPA_CAVEAT,
        "bootstrap_note": BOOTSTRAP_NOTE,
    }
    (config.RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(out, indent=2))

    print(markdown_table(overall, per_system))
    print()
    print("largest disagreements (|judge - human| desc):")
    worst = df.reindex(df.delta.abs().sort_values(ascending=False, kind="stable").index).head(10)
    for _, r in worst.iterrows():
        print(f"  {r.sheet_id} g{r.golden_id} {r.system:<6} human={r.human_overall} judge={r.judge_overall} "
              f"({r.delta:+d})  judge: {str(r.rationale)[:110]}")
    print()
    print(f"system ranking - human: {out['system_ranking']['human']}")
    print(f"system ranking - judge: {out['system_ranking']['judge']}")
    print(f"self-consistency (t=0.0 vs 0.7): {consistency.get('kappa_quadratic')} kappa, "
          f"n={consistency.get('n')}, exact={consistency.get('exact_agreement')}")


if __name__ == "__main__":
    main()
