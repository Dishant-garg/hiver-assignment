"""End-to-end evaluation on the golden set: agent + baselines for all three tasks, blinded LLM
judge on replies, and markdown/JSON outputs the report embeds. Resumable: every LLM call is cached,
so re-running after a daily cap continues where it stopped."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from support_agent import config, intents
from support_agent.agent import SupportAgent
from support_agent.baselines.escalation import AlwaysEscalate, RulesOnlyEscalation
from support_agent.baselines.intent import MajorityIntent, TfidfLogRegIntent
from support_agent.baselines.reply import CannedReply, NearestNeighbourReply
from support_agent.eval.judge import ReplyJudge
from support_agent.eval.metrics import confusion_markdown, escalation_metrics, intent_metrics
from support_agent.eval.stats import (bootstrap_mean_ci, escalation_cost_vector, macro_f1_ci,
                                      mcnemar_exact, paired_mean_diff)
from support_agent.llm import DailyLimitError, LLMClient
from support_agent.retrieve import Retriever

MISS_COST = 5.0


def _predict(golden: pd.DataFrame, corpus: pd.DataFrame, llm: LLMClient, agent_kwargs: dict) -> pd.DataFrame:
    retriever = Retriever(corpus)
    agent = SupportAgent(llm, retriever, **agent_kwargs)
    first = corpus[corpus.turn_index == 0] if "turn_index" in corpus else corpus
    maj = MajorityIntent().fit(list(first.customer_text.map(intents.weak_label)))
    try:
        lr = TfidfLogRegIntent().fit_weak(list(first.customer_text))
    except ValueError as e:
        # Degenerate corpus (too few rows or a single weak-label class, e.g. in unit tests):
        # TfidfVectorizer(min_df=2) or LogisticRegression can't fit. Never happens on the real
        # ~8000-row corpus; fall back to the majority baseline, which has the same .predict() shape.
        print(f"logreg baseline fell back to majority: fit_weak failed on a {len(first)}-row corpus ({e})", file=sys.stderr)
        lr = maj
    nn, canned = NearestNeighbourReply(retriever), CannedReply()
    always, rules_only = AlwaysEscalate(), RulesOnlyEscalation()
    rows = []
    for i, r in golden.iterrows():
        try:
            d = agent.handle(r.customer_text)
        except DailyLimitError as e:
            print(f"\nDaily cap hit after {i} rows: {e}\nRe-run later; cached rows are kept.", file=sys.stderr)
            break
        rows.append({"golden_id": r.golden_id, "customer_text": r.customer_text,
                     "agent_intent": d.intent, "agent_confidence": d.confidence, "agent_reply": d.draft_reply,
                     "agent_escalate": d.escalate, "agent_reason": d.reason, "agent_rule_hits": "|".join(d.rule_hits),
                     "agent_llm_escalate": d.llm_escalate, "agent_top_score": d.evidence[0].score if d.evidence else 0.0,
                     "agent_evidence": " || ".join(e.brand_reply for e in d.evidence),
                     "majority_intent": maj.predict([r.customer_text])[0], "logreg_intent": lr.predict([r.customer_text])[0],
                     "nn_reply": nn.reply(r.customer_text), "canned_reply": canned.reply(r.customer_text),
                     "always_escalate": always.decide(r.customer_text)[0], "rules_escalate": rules_only.decide(r.customer_text)[0]})
        print(f"\r  agent {len(rows)}/{len(golden)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return pd.DataFrame(rows)


def _judge(golden: pd.DataFrame, preds: pd.DataFrame, llm: LLMClient, systems: tuple[str, ...], canned_limit: int) -> pd.DataFrame:
    judge = ReplyJudge(llm)
    merged = preds.merge(golden[["golden_id", "reference_reply"]], on="golden_id")
    rows = []
    for sys_name in systems:
        col = {"agent": "agent_reply", "nn": "nn_reply", "canned": "canned_reply"}[sys_name]
        subset = merged.head(canned_limit) if sys_name == "canned" else merged
        for _, r in subset.iterrows():
            ev = [e for e in str(r.agent_evidence).split(" || ") if e]
            try:
                s = judge.score(r.customer_text, r.reference_reply, ev, str(r[col]))
            except DailyLimitError as e:
                print(f"\nJudge daily cap hit: {e}. Re-run later.", file=sys.stderr)
                return pd.DataFrame(rows)
            rows.append({"golden_id": r.golden_id, "system": sys_name, "candidate": r[col], **s.to_dict()})
            print(f"\r  judge {sys_name} {len(rows)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return pd.DataFrame(rows)


def _metrics(golden: pd.DataFrame, preds: pd.DataFrame, judged: pd.DataFrame) -> dict:
    g = golden.merge(preds, on="golden_id")
    y_int, y_esc = list(g.intent), [bool(v) for v in g.should_escalate]
    labels = intents.INTENT_NAMES
    out = {"n": len(g), "intent": {}, "escalation": {}, "reply": {}}
    for name, col in [("majority", "majority_intent"), ("logreg", "logreg_intent"), ("agent", "agent_intent")]:
        out["intent"][name] = intent_metrics(y_int, list(g[col]), labels)
    for name, col in [("always", "always_escalate"), ("rules_only", "rules_escalate"), ("agent", "agent_escalate")]:
        out["escalation"][name] = escalation_metrics(y_esc, [bool(v) for v in g[col]])
    if len(judged):
        for name, grp in judged.groupby("system"):
            out["reply"][name] = {k: float(grp[k].mean()) for k in ["groundedness", "correctness", "tone", "actionability", "overall"]}
            out["reply"][name]["n"] = int(len(grp))
            out["reply"][name]["share_overall_ge4"] = float((grp.overall >= 4).mean())
            out["reply"][name]["parse_failures"] = int((~grp.parse_ok.astype(bool)).sum())
    return out


def _stats(golden: pd.DataFrame, preds: pd.DataFrame, judged: pd.DataFrame) -> dict:
    """Intervals and paired tests for the numbers the report quotes.

    Exists because two of the report's comparisons turn on differences far smaller than the
    sampling noise at n=160, and a bare point estimate invites a conclusion the data cannot carry.
    Every comparison here is paired -- both systems are scored on the same rows -- so the interval
    describes the gap rather than the sum of two independent uncertainties.
    """
    g = golden.merge(preds, on="golden_id")
    labels, n = intents.INTENT_NAMES, len(g)
    y_int = list(g.intent)
    y_esc = [bool(v) for v in g.should_escalate]
    out: dict = {"n": n, "n_resamples": 10_000, "seed": config.SEED, "miss_cost": MISS_COST,
                 "intent": {}, "escalation": {}, "reply": {},
                 "note": "ci95 are seeded percentile bootstrap intervals over the golden rows. "
                         "p_a_lower is the share of resamples in which the first system's cost came out "
                         "lower (better) -- a directional bootstrap proportion, not a null-hypothesis p-value. "
                         "McNemar p_exact is a two-sided exact binomial test on the discordant rows."}

    correct = {name: np.array([t == p for t, p in zip(y_int, g[col])]) for name, col in
               [("majority", "majority_intent"), ("logreg", "logreg_intent"), ("agent", "agent_intent")]}
    for name, ok in correct.items():
        out["intent"][name] = {"accuracy_ci95": bootstrap_mean_ci(ok.astype(float)),
                               "macro_f1_ci95": macro_f1_ci(y_int, list(g[f"{name}_intent"]), labels)}
    for other in ("logreg", "majority"):
        out["intent"][f"agent_vs_{other}_mcnemar"] = mcnemar_exact(correct["agent"], correct[other])

    cost = {name: escalation_cost_vector(y_esc, [bool(v) for v in g[col]], MISS_COST) for name, col in
            [("always", "always_escalate"), ("rules_only", "rules_escalate"), ("agent", "agent_escalate")]}
    for name, c in cost.items():
        out["escalation"][name] = {"weighted_error_ci95": bootstrap_mean_ci(c)}
    for other in ("always", "rules_only"):
        out["escalation"][f"agent_vs_{other}"] = paired_mean_diff(cost["agent"], cost[other])

    if len(judged):
        wide = judged.pivot(index="golden_id", columns="system", values="overall")
        for name in wide.columns:
            out["reply"][name] = {"overall_ci95": bootstrap_mean_ci(wide[name].dropna().values.astype(float))}
        for other in [c for c in wide.columns if c != "agent"]:
            if "agent" not in wide.columns:
                break
            both = wide[["agent", other]].dropna()
            # Higher judge score is better, so the diff is signed the other way round from cost:
            # p_a_lower here is the share of resamples where the agent scored *worse*.
            if len(both):
                out["reply"][f"agent_vs_{other}"] = paired_mean_diff(both["agent"].values.astype(float),
                                                                     both[other].values.astype(float))
    return out


def _ci(s: dict | None, key: str) -> str:
    if not s or key not in s:
        return "n/a"
    lo, hi = s[key]
    return f"[{lo:.3f}, {hi:.3f}]"


def _summary_md(m: dict) -> str:
    st = m.get("stats", {})
    si, se, sr = st.get("intent", {}), st.get("escalation", {}), st.get("reply", {})
    L = ["# Evaluation summary", f"n = {m['n']} golden examples", "",
         "95% CIs are seeded percentile bootstraps over the golden rows "
         f"({st.get('n_resamples', 0):,} resamples, seed {st.get('seed', '?')}). They describe sampling "
         "noise at this n only -- not labelling error, and not the fact that this is one brand.", "",
         "## Intent", "| system | accuracy | 95% CI | macro-F1 | 95% CI |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['accuracy']:.3f} | {_ci(si.get(k), 'accuracy_ci95')} | {v['macro_f1']:.3f} | {_ci(si.get(k), 'macro_f1_ci95')} |"
          for k, v in m["intent"].items()]
    L += ["", "## Escalation (positive = escalate)",
          "| system | precision | recall | missed-esc rate | unnecessary-esc rate | automation rate | weighted error | 95% CI |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['missed_escalation_rate']:.3f} | {v['unnecessary_escalation_rate']:.3f} | {v['automation_rate']:.3f} | {v['weighted_error']:.3f} | {_ci(se.get(k), 'weighted_error_ci95')} |"
          for k, v in m["escalation"].items()]
    if m["reply"]:
        L += ["", "## Reply quality (LLM judge, 1-5)",
              "| system | n | grounded | correct | tone | actionable | overall | overall 95% CI | share overall>=4 | parse fail |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['groundedness']:.2f} | {v['correctness']:.2f} | {v['tone']:.2f} | {v['actionability']:.2f} | {v['overall']:.2f} | {_ci(sr.get(k), 'overall_ci95')} | {v['share_overall_ge4']:.2f} | {v['parse_failures']} |"
              for k, v in m["reply"].items()]
    if st:
        L += ["", "## Head-to-head, paired on the same rows", "",
              "Each comparison scores both systems on identical rows, so the interval is on the *gap*.", ""]
        for other in ("logreg", "majority"):
            t = si.get(f"agent_vs_{other}_mcnemar")
            if t:
                L.append(f"- **Intent, agent vs {other}** (exact McNemar): agent alone correct on {t['b']} rows, "
                         f"{other} alone correct on {t['c']}, p = {t['p_exact']:.2e}.")
        for other in ("always", "rules_only"):
            d = se.get(f"agent_vs_{other}")
            if d:
                L.append(f"- **Escalation weighted error, agent minus {other}**: {d['diff']:+.4f} "
                         f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]. Agent came out cheaper in "
                         f"{d['p_a_lower']:.0%} of resamples (lower cost is better).")
        for other in ("nn", "canned"):
            d = sr.get(f"agent_vs_{other}")
            if d:
                L.append(f"- **Judged overall, agent minus {other}** (n = {d['n']}): {d['diff']:+.3f} "
                         f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]. Agent scored lower in "
                         f"{d['p_a_lower']:.0%} of resamples (higher score is better).")
    return "\n".join(L) + "\n"


def run_eval(golden_path: Path, out_dir: Path, *, corpus_path: Path | None = None, llm: LLMClient | None = None,
             limit: int | None = None, skip_judge: bool = False, judge_systems=("agent", "nn", "canned"),
             canned_judge_limit: int = 40, agent_kwargs: dict | None = None) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    golden = pd.read_csv(golden_path)
    if limit:
        golden = golden.head(limit)
    corpus = pd.read_csv(corpus_path or config.brand_dir() / "corpus.csv")
    llm = llm or LLMClient()
    preds = _predict(golden, corpus, llm, agent_kwargs or {})
    if not len(preds):
        print("Daily cap hit before any row was predicted; nothing written. Re-run later.", file=sys.stderr)
        return {}
    preds.to_csv(out_dir / "predictions.csv", index=False)
    judged = pd.DataFrame() if skip_judge else _judge(golden, preds, llm, judge_systems, canned_judge_limit)
    if len(judged):
        judged.to_csv(out_dir / "judge_scores.csv", index=False)
    golden = golden[golden.golden_id.isin(preds.golden_id)]
    m = _metrics(golden, preds, judged)
    m["stats"] = _stats(golden, preds, judged)
    m["llm_cache"] = dict(llm.stats)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    (out_dir / "summary.md").write_text(_summary_md(m))
    (out_dir / "intent_confusion.md").write_text("# Agent intent confusion\n\n" + confusion_markdown(m["intent"]["agent"]) + "\n")
    print(_summary_md(m))
    return m


def _guard_partial_overwrite(out: Path, limit: int | None, skip_judge: bool, force: bool) -> None:
    """Refuse to overwrite the committed `results/` with a partial run.

    `--limit 20` and `--skip-judge` both produce a results directory that looks complete but
    is not: the first reports metrics on a fraction of the golden set, the second silently drops
    the reply-quality table. Pointed at the default `--out`, either one quietly replaces the
    committed headline numbers with weaker ones. Writing elsewhere is always allowed; clobbering
    the real results with a partial run takes `--force`.
    """
    if force or not (limit or skip_judge):
        return
    if out.resolve() != config.RESULTS_DIR.resolve():
        return
    flags = " and ".join(f for f in (f"--limit {limit}" if limit else "", "--skip-judge" if skip_judge else "") if f)
    raise SystemExit(
        f"Refusing to overwrite {out} with a partial run ({flags}).\n"
        f"That directory holds the committed headline results, and a partial run would replace them "
        f"with metrics on fewer rows or with no reply-quality table.\n"
        f"Write somewhere else, e.g. --out results_smoke, or pass --force if you really mean it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default=str(config.GOLDEN_DIR / "golden_set.csv"))
    p.add_argument("--out", default=str(config.RESULTS_DIR))
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--force", action="store_true", help="allow a partial run to overwrite the committed results/")
    a = p.parse_args()
    _guard_partial_overwrite(Path(a.out), a.limit, a.skip_judge, a.force)
    run_eval(Path(a.golden), Path(a.out), limit=a.limit, skip_judge=a.skip_judge)
