# AI customer-support agent for SpotifyCares

A support agent for a single Twitter brand, built on the [Customer Support on Twitter][twcs]
dataset, together with the evaluation needed to decide whether it can be trusted.

Given one incoming customer message, the agent retrieves similar exchanges from the brand's own
reply history, makes a single structured LLM call that classifies the intent and drafts a reply,
and then decides whether that draft can be sent or a human must take over.

**The escalation decision is the point of the project.** Deterministic rules run on both sides of
the model, the model's own request for a human is an input to that decision rather than the final
word, and every escalation carries a reason an agent can act on. Drafting a plausible tweet is the
easy half; knowing when not to send one is the half worth building carefully.

The system is measured against 160 hand-labelled held-out messages, with two baselines on each of
the three tasks, an LLM judge for reply quality, and an agreement study that bounds how far that
judge can be trusted.

| | |
|---|---|
| **Brand** | SpotifyCares, chosen by a measured survey of the twelve highest-volume brands ([decision log 1](docs/07-decision-log.md)) |
| **Inference** | `openai/gpt-oss-120b` on Groq's free tier, temperature 0 |
| **Retrieval** | TF-IDF over 8,000 historical customer/brand pairs |
| **Evaluation** | 160 hand-labelled messages, 6 baselines, blinded LLM judge, bootstrap intervals |

```bash
make setup && make verify     # ~100 s from a clean clone, no API key required
```

[twcs]: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

---

## Results

All figures are `results/summary.md` verbatim, on n = 160 held-out messages. The full write-up is
[`docs/01-report.md`](docs/01-report.md); the individual cases behind it are in
[`results/failures.md`](results/failures.md).

**Intent classification**

| system | accuracy | 95% CI | macro-F1 | 95% CI |
|---|---|---|---|---|
| majority class | 0.125 | [0.075, 0.181] | 0.022 | [0.014, 0.031] |
| TF-IDF + logistic regression | 0.463 | [0.388, 0.537] | 0.463 | [0.364, 0.542] |
| **agent** | **0.744** | [0.675, 0.806] | **0.685** | [0.599, 0.761] |

**Escalation** (positive = escalate; `weighted error` prices a missed escalation at 5x an
unnecessary one)

| system | precision | recall | missed | unnecessary | automation | weighted error | 95% CI |
|---|---|---|---|---|---|---|---|
| always escalate | 0.487 | 1.000 | 0.000 | 0.512 | 0.000 | 0.512 | [0.438, 0.588] |
| rules only | 1.000 | 0.103 | 0.438 | 0.000 | 0.950 | 2.188 | [1.812, 2.562] |
| **agent** | 0.636 | 0.872 | 0.062 | 0.244 | **0.331** | 0.556 | [0.381, 0.756] |

**Reply quality** (LLM judge, 1-5)

| system | n | grounded | correct | tone | actionable | overall | 95% CI | >=4 | parse fail |
|---|---|---|---|---|---|---|---|---|---|
| **agent** | 160 | 3.24 | 3.27 | 4.51 | 3.33 | **3.07** | [2.83, 3.31] | 0.46 | 0 |
| nearest neighbour | 160 | 4.95 | 2.64 | 4.38 | 2.96 | 2.66 | [2.40, 2.93] | 0.38 | 0 |
| canned deflection | 40 | 2.85 | 2.10 | 3.83 | 2.27 | 2.12 | [1.83, 2.45] | 0.15 | 0 |

### What the numbers support

A point estimate on 160 rows can easily imply more than the data carries, so every comparison is
also run **paired** — both systems scored on the same rows, bootstrapping the gap rather than each
system separately.

- **Intent, agent vs. logistic regression.** The agent is alone correct on 59 rows, the baseline on
  14. Exact McNemar *p* = 1.0e-07. This one is not close.
- **Reply quality, agent vs. nearest neighbour.** +0.406 [+0.113, +0.694], ahead in 100% of
  resamples. The per-system intervals overlap, which is precisely why the pairing matters.
- **Escalation, agent vs. always-escalate.** +0.0437 [-0.156, +0.263], with the agent cheaper in
  34% of resamples. **The two are indistinguishable at this sample size** — the interval is nearly
  ten times the width of the gap inside it.

So the agent leads clearly on intent and on judged reply quality, and it handles 33.1% of messages
without a human. On cost-weighted safety it is *not measurably different* from escalating
everything: it misses 10 of 78 true escalations, which always-escalate by construction cannot do,
while always-escalate automates nothing, which the agent's 33.1% cannot be bought back from at any
price. Sections 4 and 7 of the report take that apart, including the `miss_cost` constant the whole
comparison rests on.

Thresholds were tuned on the 40-row dev slice and never on the golden set. [Decision log
7](docs/07-decision-log.md) records the grid, the trade it bought, and two caveats worth knowing:
the confidence gate turned out inert, and the dev slice contains no `app_bug` row.

---

## Reproducing this

```bash
make setup     # create .venv, install the package pinned to constraints.txt
make verify    # replay the committed cache offline, assert results/ is unchanged
make test      # 88 tests, all offline
```

From a clean clone this takes about 100 seconds end to end, against the assignment's 15-minute
budget, and needs no API key.

`make reproduce` replays the evaluation from `data/cache/llm_cache.jsonl` with `LLM_OFFLINE=1`,
which turns any prompt missing from the cache into a `CacheMissError` rather than a silent live
call — so the run either reproduces exactly or fails loudly. It replays all 520 responses and
recomputes every bootstrap interval in under ten seconds. `make verify` does the same and then
diffs the output, so it fails if anything moved; the bootstraps are seeded from `config.SEED`, so
the intervals reproduce byte-for-byte alongside the metrics.

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the tests, the reproduction, and a
check that the partial-run guard fires — all with no API key in the environment. That is what makes
the offline claim checkable rather than merely asserted.

### Running against the live API

```bash
cp .env.example .env    # add your GROQ_API_KEY
make data               # download twcs.csv, rebuild corpus.csv + holdout.csv (~30 s)
make golden             # re-sample the unlabelled golden/dev sheets (labels are committed)
make eval               # agent + six baselines + blinded judge -> results/
make demo MSG="my playlist vanished after the update"
```

Useful flags on the harness:

```bash
python -m support_agent.eval.run --skip-judge --out results_tmp   # metrics only, no judge calls
python -m support_agent.eval.run --limit 20 --out results_smoke   # quick smoke run
python -m support_agent.eval.tune                                 # thresholds, dev slice only
```

`--limit` and `--skip-judge` each produce a results directory that looks complete but is not: one
reports metrics on a fraction of the golden set, the other silently drops the reply-quality table.
Pointed at the default output, either would quietly replace the committed headline numbers, so the
harness refuses to overwrite `results/` with a partial run. Pass a different `--out`, or `--force`
if you genuinely mean it.

> **Working within the free tier.** Groq caps tokens per day per model at roughly 200K. A full
> golden run is 160 agent calls plus about 360 judge calls, and the judge model has its own budget.
> Three mitigations are built in: every response is cached by a hash of its exact inputs, so
> nothing is paid for twice; calls are spaced by a fixed minimum interval for the per-minute limit;
> and a per-day 429 raises `DailyLimitError`, which the harness catches to stop cleanly and keep
> every row it already has. Expect a first full run to need two sittings.

---

## The golden set

200 messages were sampled from the **time-held-out** slice alone, so nothing the agent is scored on
sits in the index it searches. The corpus is the earlier 75% of calendar days and the holdout the
later 25%, split by day rather than at random so that no thread — and no same-day boilerplate —
straddles the line. Sampling takes first turns only and stratifies on a keyword weak label with a
floor per stratum, because a uniform draw would have been dominated by two intents and left the
rare ones unmeasurable.

Every row was then labelled by hand against [`data/golden/labeling-guide.md`](data/golden/labeling-guide.md),
a guide fixed before labelling began, reading both the customer message and the brand's real reply,
in batches of 40, with a written justification on all 200 rows. The weak labeller disagreed on
45.6% of them, which is the argument for labelling by hand. The result is `golden_set.csv` (160
rows, the evaluation ground truth) and `dev_set.csv` (40 rows, disjoint by `pair_id`, the only
slice thresholds may be tuned on).

The set was audited against the guide rather than trusted, and the audit ships as tests in
[`tests/test_golden_integrity.py`](tests/test_golden_integrity.py): no duplicate or leaked row,
zero overlap with the retrieval corpus on `pair_id`, tweet id or verbatim text, the time split
intact, and every never-auto-handle intent escalating without exception.

**Two caveats before quoting any figure from it.** The intent distribution is
stratify-then-relabel and is *not* a prevalence estimate. And the labels come from a single
AI-assisted labeller, with no inter-annotator agreement figure. Both, along with everything else
worth distrusting, are set out in [`docs/03-golden-set.md`](docs/03-golden-set.md).

---

## Documentation

`docs/` is numbered in reading order; [`docs/00-index.md`](docs/00-index.md) is the full index.

| # | Document | What it answers |
|---|---|---|
| 01 | [Report](docs/01-report.md) | **The write-up.** Framing, results against two baselines per task, the top five failure modes, what is misleading about the headline number, and what one more week would buy. |
| 02 | [Architecture](docs/02-architecture.md) | Both diagrams, and a paragraph per component. |
| 03 | [Golden set](docs/03-golden-set.md) | Sampling, labelling, distribution, hardest calls, audit, limitations. |
| 04 | [Judge rubric](docs/04-judge-rubric.md) | The rubric, how blinding is enforced, the known biases. |
| 05 | [Judge agreement](docs/05-judge-agreement.md) | How far the judge can be trusted, and where it cannot. |
| 06 | [Intent taxonomy](docs/06-intent-taxonomy.md) | The taxonomy, and why each auto-handle flag is what it is. |
| 07 | [Decision log](docs/07-decision-log.md) | The non-obvious decisions, each with what it cost. |
| 08 | [Code walkthrough](docs/08-code-walkthrough.md) | Every file, a reading order, and "how to change X" recipes. |
| 09 | [Citations](docs/09-citations.md) | Dataset, models, libraries, papers, and AI assistance. |

---

## Repository layout

```text
src/support_agent/
  config.py            paths, model ids, thresholds, seeds; reads .env
  intents.py           9 intents + other, keyword weak labeller, prompt block
  rules.py             pre_check on the message, post_check on the model output
  retrieve.py          TF-IDF retriever over the corpus, cosine top-k
  llm.py               the only module that calls an LLM; cache, rate spacer, daily cap
  agent.py             prompt, JSON parsing, AgentDecision, SupportAgent.handle
  cli.py               python -m support_agent.cli demo "message"
  data/                download, build_pairs, sample_golden
  baselines/           intent, reply, escalation — trivial and simple for each task
  eval/                metrics, stats, judge, agreement, run, tune, failures

data/
  processed/           corpus.csv (8,000 pairs), holdout.csv (23,429 pairs)
  golden/              golden_set.csv (160), dev_set.csv (40), labeling-guide.md
  cache/               llm_cache.jsonl — 602 responses, committed so reproduction needs no key

results/               written by make eval / make reproduce
scripts/               one-off analyses quoted in the decision log
tests/                 17 files, 88 tests, all offline
docs/                  numbered in reading order
```

---

## Assignment deliverables

| Deliverable | Where |
|---|---|
| Runnable pipeline, reproducible in under 15 minutes | `make setup && make verify` — ~100 s, no API key |
| Golden set of 150-250 hand-labelled examples | `data/golden/golden_set.csv` (160) + `dev_set.csv` (40) |
| Note on how it was sampled and labelled | [`docs/03-golden-set.md`](docs/03-golden-set.md), rules in [`labeling-guide.md`](data/golden/labeling-guide.md) |
| Evaluation harness with automated metrics | [`eval/run.py`](src/support_agent/eval/run.py), [`metrics.py`](src/support_agent/eval/metrics.py), intervals in [`stats.py`](src/support_agent/eval/stats.py) |
| LLM-as-judge with a rubric | [`eval/judge.py`](src/support_agent/eval/judge.py), documented in [`docs/04-judge-rubric.md`](docs/04-judge-rubric.md) |
| Evidence the judge agrees with a human | [`eval/agreement.py`](src/support_agent/eval/agreement.py), write-up in [`docs/05-judge-agreement.md`](docs/05-judge-agreement.md) |
| Two baselines per task | [`baselines/`](src/support_agent/baselines) — trivial and simple for intent, reply and escalation |
| Report, at most 6 pages | [`docs/01-report.md`](docs/01-report.md) |
| Failure analysis | [`docs/01-report.md`](docs/01-report.md) section 6, cases in [`results/failures.md`](results/failures.md) |
| "What is misleading about my headline number" | [`docs/01-report.md`](docs/01-report.md) section 7 |
| Decision log | [`docs/07-decision-log.md`](docs/07-decision-log.md) |

## Licence

MIT — see [LICENSE](LICENSE).
