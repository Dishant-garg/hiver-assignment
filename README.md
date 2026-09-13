# AI customer-support agent for SpotifyCares

An AI support agent for one Twitter brand, built on the Customer Support on Twitter dataset. It
takes a single incoming customer message, retrieves similar historical exchanges from the brand's
own reply history, makes one structured LLM call that classifies the intent and drafts a reply,
and then decides whether the draft can be sent or a human must take over. The escalation decision
is the point of the project: deterministic rules run before and after the model, the model's own
request for a human is one input rather than the final word, and every escalation carries a reason
a human can act on. The whole thing is evaluated against 160 hand-labelled held-out messages, with
two baselines on each of the three tasks, an LLM judge for reply quality, and an agreement study
that says how far that judge can be trusted.

Brand: **SpotifyCares**, chosen by a measured survey of the twelve highest-volume brands
(`docs/07-decision-log.md` entry 1). Inference: Groq's free tier. Retrieval: TF-IDF over 8,000
historical pairs.

## Headline results

n = 160 hand-labelled held-out messages. Tables are `results/summary.md` verbatim; the full
write-up is [`docs/01-report.md`](docs/01-report.md) and the cases behind it are `results/failures.md`.

### Intent
| system | accuracy | 95% CI | macro-F1 | 95% CI |
|---|---|---|---|---|
| majority | 0.125 | [0.075, 0.181] | 0.022 | [0.014, 0.031] |
| logreg | 0.463 | [0.388, 0.537] | 0.463 | [0.364, 0.542] |
| agent | 0.744 | [0.675, 0.806] | 0.685 | [0.599, 0.761] |

### Escalation (positive = escalate)
| system | precision | recall | missed-esc rate | unnecessary-esc rate | automation rate | weighted error | 95% CI |
|---|---|---|---|---|---|---|---|
| always | 0.487 | 1.000 | 0.000 | 0.512 | 0.000 | 0.512 | [0.438, 0.588] |
| rules_only | 1.000 | 0.103 | 0.438 | 0.000 | 0.950 | 2.188 | [1.812, 2.562] |
| agent | 0.636 | 0.872 | 0.062 | 0.244 | 0.331 | 0.556 | [0.381, 0.756] |

### Reply quality (LLM judge, 1-5)
| system | n | grounded | correct | tone | actionable | overall | overall 95% CI | share overall>=4 | parse fail |
|---|---|---|---|---|---|---|---|---|---|
| agent | 160 | 3.24 | 3.27 | 4.51 | 3.33 | 3.07 | [2.825, 3.312] | 0.46 | 0 |
| canned | 40 | 2.85 | 2.10 | 3.83 | 2.27 | 2.12 | [1.825, 2.450] | 0.15 | 0 |
| nn | 160 | 4.95 | 2.64 | 4.38 | 2.96 | 2.66 | [2.400, 2.925] | 0.38 | 0 |

Point estimates alone would overstate what n=160 can settle, so every comparison is also run
paired -- both systems scored on the same rows, bootstrapped on the gap rather than on each system
separately:

- **Intent, agent vs logreg**: agent alone correct on 59 rows, logreg alone on 14. Exact McNemar
  p = 1.0e-07. The agent is better; this one is not close.
- **Reply quality, agent vs nearest-neighbour**: +0.406 [+0.113, +0.694]. The interval excludes
  zero, and the agent won in 100% of resamples.
- **Escalation, agent vs always-escalate**: +0.0437 [-0.156, +0.263], agent cheaper in 34% of
  resamples. **The two are indistinguishable at this sample size.** The interval is nearly ten
  times the width of the gap inside it, so neither "the agent wins" nor "the agent loses" is a
  finding.

So the agent clearly leads on intent and on judged reply quality, and it automates 33.1% of
messages. What the escalation table shows is that **its cost-weighted safety is not measurably
different from escalating everything** -- it misses 10 of 78 true escalations, which
always-escalate cannot do, but always-escalate automates nothing, and n=160 cannot separate the
two. `docs/01-report.md` sections 4 and 7 treat that honestly, including the `miss_cost` constant the
comparison rests on.

The evaluation has been run live against Groq: `results/` holds the artifacts and
`data/cache/llm_cache.jsonl` is committed so `make reproduce` replays them offline with no key.
The escalation thresholds in `src/support_agent/config.py` were tuned on the 40-row dev slice,
never on the golden set — `docs/07-decision-log.md` entry 7 records the grid, the recall/automation
trade-off it bought, and the two caveats worth knowing (the confidence gate turned out inert, and
the dev slice contains no `app_bug` row). The judge model id is verified against the live
`/models` list in entry 5.

## Quickstart

```bash
make setup       # creates .venv, installs the package pinned to constraints.txt
make verify      # offline replay + asserts results/ is byte-identical; no API key needed
make test        # 80 tests, all offline, no network calls
make demo MSG="my playlist vanished after the update"   # one message, needs a key
```

`make reproduce` sets `LLM_OFFLINE=1`, which makes any prompt that is not already in
`data/cache/llm_cache.jsonl` raise `CacheMissError` instead of calling out. The run either
reproduces exactly or fails loudly. It rewrites `results/` in place, and the cache hit/miss
counter in `metrics.json` is the only value that changes. It replays all 520 cached responses and recomputes every
bootstrap interval in **under 10 seconds**, so the whole check is well under two minutes after
`make setup`, against an assignment budget of 15 minutes. Its `summary.md` is byte-identical to the
committed one: the bootstraps are seeded from `config.SEED`, so the intervals reproduce exactly
too.

`make reproduce` does the replay and rewrites `results/`; `make verify` does the same and then
diffs, so it fails if anything moved. `.github/workflows/ci.yml` runs `pytest`, `verify`, and a
check that the partial-run guard fires — all with no API key in the environment, which is what
makes the offline claim checkable rather than asserted.

`make demo` calls the live API, so it needs a key. Everything else in the quickstart is offline.

## Running it live

A live run needs a Groq API key. The free tier is enough, but it is a real constraint.

```bash
cp .env.example .env      # then paste your key into GROQ_API_KEY
make data                 # download twcs.csv and rebuild corpus.csv + holdout.csv (~30s)
make golden               # re-sample the unlabelled golden/dev sheets (labels are committed)
make eval                 # full evaluation: agent, six baselines, blinded judge -> results/
```

Useful flags on the harness itself:

```bash
.venv/bin/python -m support_agent.eval.run --skip-judge --out results_tmp   # metrics, no judge calls
.venv/bin/python -m support_agent.eval.run --limit 20 --out results_smoke
.venv/bin/python -m support_agent.eval.tune                             # thresholds, dev slice only
```

`--limit` and `--skip-judge` both produce a results directory that looks complete but is not: one
reports metrics on a fraction of the golden set, the other silently drops the reply-quality table.
Pointed at the default `--out`, either would quietly replace the committed headline numbers, so the
harness now **refuses** to overwrite `results/` with a partial run and tells you to pick another
`--out`. `--force` is the deliberate override.

**Daily budget.** Groq's free tier caps tokens per day per model, in the region of 200K. A full
golden run is 160 agent calls plus roughly 360 judge calls, and the judge model has its own
separate budget. Three things follow, and all three are built in. Every response is cached by a
hash of its exact inputs, so nothing is ever paid for twice. Calls are spaced by a fixed minimum
interval for the per-minute limit. A per-day 429 raises `DailyLimitError`, which the harness
catches to stop cleanly, keeping every row it already has, so re-running tomorrow continues from
the cache instead of starting over. Expect a first full run to need two sittings.

## How the golden set was built

200 messages were sampled from the **time-held-out** slice only, so nothing the agent is scored on
is in the index it searches. The corpus is the earlier 75% of calendar days and the holdout is the
later 25%, split by day rather than randomly so no thread or same-day boilerplate straddles the
line. Sampling takes first turns only and stratifies by a keyword weak label, with a floor per
stratum, because a uniform sample would have been dominated by two intents and would have left the
rare ones unmeasurable.

Every row was then labelled by hand against `data/golden/labeling-guide.md`, a guide fixed before
labelling started, reading both the customer message and the brand's real reply, in batches of 40,
with a written justification on every one of the 200 rows. The weak label was wrong on 45.6% of
golden rows, which is the point of labelling by hand. The result is `golden_set.csv` (160 rows,
the evaluation ground truth) and `dev_set.csv` (40 rows, disjoint by `pair_id`, the only slice
thresholds may be tuned on).

Two things to know before quoting any number from it. The intent distribution is
stratify-then-relabel and is **not** a prevalence estimate. And the labels come from a single
AI-assisted labeller with no inter-annotator agreement figure. Both are stated plainly, with
everything else worth distrusting, in `docs/03-golden-set.md`.

## Repository layout

```text
README.md                     this file
Makefile                      setup / data / golden / eval / reproduce / verify / test / demo
pyproject.toml                dependencies and package metadata
constraints.txt               exact versions the committed results were produced with
.github/workflows/ci.yml      tests + byte-identical reproduction, run with no API key
.env.example                  template; only GROQ_API_KEY is required, and only for live runs

src/support_agent/
  config.py                   paths, model ids, thresholds, seeds; reads .env
  intents.py                  9 intents + other, keywords, weak labeller, prompt block
  rules.py                    pre_check on the message, post_check on the model output
  retrieve.py                 TF-IDF retriever over the corpus, cosine top-k
  llm.py                      the only module that calls an LLM; cache, rate spacer, daily cap
  agent.py                    prompt, JSON parsing, AgentDecision, SupportAgent.handle
  cli.py                      python -m support_agent.cli demo "message"
  data/                       download.py, build_pairs.py, sample_golden.py
  baselines/                  intent.py, reply.py, escalation.py (trivial + simple, each task)
  eval/                       metrics.py, stats.py, judge.py, agreement.py, run.py, tune.py, failures.py

scripts/                      survey_brands.py, explore_intents.py, intent_deflection.py, judge_consistency.py
tests/                        17 test files, all offline
data/processed/SpotifyCares/  corpus.csv (8,000 pairs), holdout.csv (23,429 pairs)
data/golden/                  golden_set.csv (160), dev_set.csv (40), labeling-guide.md
data/cache/                   llm_cache.jsonl, committed so reproduction needs no key
results/                      written by make eval / make reproduce; failures.md by eval.failures
docs/                         00-index, 01-report, 02-architecture, ... numbered in reading order
```

## Documents

`docs/` is numbered in reading order; [`docs/00-index.md`](docs/00-index.md) is the full index.

| # | Document | What it answers |
|---|---|---|
| 01 | [`docs/01-report.md`](docs/01-report.md) | **The report.** Framing, results against two baselines per task, top-five failure modes, what is misleading about the headline number, and what one more week would buy. |
| 02 | [`docs/02-architecture.md`](docs/02-architecture.md) | Both diagrams, and one paragraph per component. |
| 03 | [`docs/03-golden-set.md`](docs/03-golden-set.md) | Sampling, labelling, distribution, hardest calls, limitations. |
| 04 | [`docs/04-judge-rubric.md`](docs/04-judge-rubric.md) | The rubric, the blinding, the known biases. |
| 05 | [`docs/05-judge-agreement.md`](docs/05-judge-agreement.md) | Judge-versus-human agreement, and the verdict on trusting the judge. |
| 06 | [`docs/06-intent-taxonomy.md`](docs/06-intent-taxonomy.md) | The taxonomy, and why each auto-handle flag is what it is. |
| 07 | [`docs/07-decision-log.md`](docs/07-decision-log.md) | The non-obvious decisions, each with its cost. |
| 08 | [`docs/08-code-walkthrough.md`](docs/08-code-walkthrough.md) | Every file, a reading order, and "how to change X" recipes. |
| 09 | [`docs/09-citations.md`](docs/09-citations.md) | Dataset, models, libraries, papers, AI assistance. |

## Submission checklist

| Assignment deliverable | Where it is |
|---|---|
| A repository with a runnable pipeline, reproducible in under 15 minutes | `make setup && make reproduce`, driven by `Makefile` and `src/support_agent/eval/run.py`, replaying `data/cache/llm_cache.jsonl` with no API key |
| Golden evaluation set of 150-250 hand-labelled examples | `data/golden/golden_set.csv` (160 rows) plus `data/golden/dev_set.csv` (40 rows), 200 hand-labelled in total |
| A note on how it was sampled and labelled | `docs/03-golden-set.md`, with the rules in `data/golden/labeling-guide.md` and the sampler in `src/support_agent/data/sample_golden.py` |
| Evaluation harness with automated metrics | `src/support_agent/eval/run.py` and `src/support_agent/eval/metrics.py`, with bootstrap CIs and paired tests in `src/support_agent/eval/stats.py`; output in `results/summary.md`, `results/metrics.json`, `results/intent_confusion.md` |
| LLM-as-judge with a rubric | `src/support_agent/eval/judge.py`, documented in `docs/04-judge-rubric.md`; scores in `results/judge_scores.csv` |
| Judge-versus-human agreement evidence | `src/support_agent/eval/agreement.py` and `scripts/judge_consistency.py`; write-up in `docs/05-judge-agreement.md` |
| Two baselines per task | `src/support_agent/baselines/intent.py`, `reply.py`, `escalation.py`: trivial and simple for intent, reply and escalation |
| Report, at most 6 pages | `docs/01-report.md` (~6 pages rendered) |
| Problem framing | `docs/01-report.md` section 1, with the scope boundaries it follows from recorded in `docs/07-decision-log.md` |
| Results against the baselines | `docs/01-report.md` section 4, from `results/summary.md` |
| Top-five failure analysis | `docs/01-report.md` section 6, from `results/failures.md` (regenerate with `python -m support_agent.eval.failures`) |
| "What is misleading about my headline number" | `docs/01-report.md` section 7 |
| What one more week would buy | `docs/01-report.md` section 8 |
| Decision log of 10-15 non-obvious decisions | `docs/07-decision-log.md`, 16 entries |
