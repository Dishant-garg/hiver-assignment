# Code walkthrough

How to navigate this repository, in what order to read it, and how to change the things a
reviewer is most likely to want changed. Paths are relative to the repository root.

## 1. Every file

### Project root

| File | What it is |
|---|---|
| `00-index.md` | Quickstart, results, repo layout, submission checklist. |
| `Makefile` | The seven commands: `setup`, `data`, `golden`, `eval`, `reproduce`, `test`, `demo`. |
| `pyproject.toml` | Package metadata and dependencies. `pandas`, `pyarrow`, `scikit-learn`, `openai`, `python-dotenv`, `pydantic`, `tabulate`, plus `pytest` as the dev extra. Sources live under `src/`. |
| `.env.example` | Template for `.env`. Only `GROQ_API_KEY` is required, and only for live runs. |
| `.gitignore` | Keeps `.env`, `.venv/`, `data/raw/` and caches out of the repository. |

### `src/support_agent/` (the package)

| File | What it is |
|---|---|
| `__init__.py` | Empty. Marks the package. |
| `config.py` | Every path, model id, threshold, seed and size constant. Loads `.env`. Exposes `brand_dir()` and `offline()`. |
| `intents.py` | The taxonomy: nine intents plus `other`, each with a definition, examples, keywords and an `auto_handle_allowed` flag. Also the keyword weak labeller and the prompt's intent block. |
| `rules.py` | `pre_check` on the raw message, `post_check` on the model output, and `ESCALATION_CATEGORIES`, the vocabulary the golden set's `escalation_reason` column must use. |
| `retrieve.py` | `Retriever` fits TF-IDF over the corpus and returns the top-k most similar past exchanges with a cosine score. |
| `llm.py` | `LLMClient`: the persistent JSONL cache, the rate spacer, the Groq transport, `CacheMissError` and `DailyLimitError`. |
| `agent.py` | The prompt, the JSON parser, `AgentDecision`, and `SupportAgent.handle`, which is the whole runtime path in fifteen lines. |
| `cli.py` | `python -m support_agent.cli demo "message"`. Prints one decision as JSON. |

### `src/support_agent/data/` (building the dataset)

| File | What it is |
|---|---|
| `__init__.py` | Empty. |
| `download.py` | Fetches `twcs.csv` from Kaggle's public endpoint, falling back to a Hugging Face mirror. No-op if the file exists. |
| `build_pairs.py` | Cleaning, pair reconstruction, `turn_index`, and the by-day 75/25 time split. Writes `corpus.csv` and `holdout.csv`. |
| `sample_golden.py` | Stratified sampling of first-turn holdout messages into the unlabelled golden and dev sheets. |

### `src/support_agent/baselines/` (what the agent must beat)

| File | What it is |
|---|---|
| `__init__.py` | Empty. |
| `intent.py` | `MajorityIntent` (trivial) and `TfidfLogRegIntent` (simple, trained on weak labels only). |
| `reply.py` | `CannedReply` (trivial) and `NearestNeighbourReply` (simple, returns the closest historical brand reply verbatim). |
| `escalation.py` | `AlwaysEscalate` (trivial) and `RulesOnlyEscalation` (simple, the rules layer with no model). |

### `src/support_agent/eval/` (measuring it)

| File | What it is |
|---|---|
| `__init__.py` | Empty. |
| `metrics.py` | Intent metrics with a confusion matrix, escalation metrics with the cost-weighted error, and a markdown confusion table. |
| `stats.py` | Uncertainty for those metrics: seeded bootstrap CIs, paired bootstrap of the gap between two systems, and an exact McNemar test. `escalation_cost_vector` is the trick that makes it simple — it expresses the cost-weighted error as a per-row cost, so both the interval and the paired comparison reduce to bootstrapping a mean. |
| `judge.py` | The 1-5 rubric, the blinded judge prompt, score clamping, and the `parse_ok` flag. |
| `agreement.py` | Quadratic-weighted kappa, Spearman rho, exact and within-one agreement. |
| `run.py` | The end-to-end harness: predict, judge, score, write `results/`. Has the `--golden`, `--out`, `--limit` and `--skip-judge` flags. |
| `tune.py` | Grid search over the two post-check thresholds on the dev slice only. |

### `scripts/` (one-off analyses, read by a person)

| File | What it is |
|---|---|
| `survey_brands.py` | The brand comparison table in decision log entry 1: pair volume, reply length, deflection share, resolution-step share, non-English share. |
| `intent_deflection.py` | The per-intent `dm_deflect%` / `resolution_step%` table in `docs/06-intent-taxonomy.md`: weak-labels the first-turn corpus rows and applies the survey's deflection-any and resolution-step regexes. |
| `explore_intents.py` | TF-IDF plus KMeans over first-turn messages, printing cluster terms and samples. Used to derive the taxonomy. |
| `judge_consistency.py` | Re-scores forty agent replies at temperature 0.0 and 0.7 and reports agreement between the two passes. |

### `tests/` (all offline, no network)

| File | What it is |
|---|---|
| `__init__.py` | Empty. |
| `fixtures/mini_twcs.csv` | A handful of rows shaped like the raw dump, for the pair builder tests. |
| `test_config.py` | Paths stay under the repository root; `LLM_OFFLINE` is read correctly. |
| `test_intents.py` | Taxonomy shape, the escalate-always flags, weak labelling, nested keywords counted once, prompt block size. |
| `test_rules.py` | Each pre rule fires on a real example and stays quiet on a routine complaint; post-check thresholds; the promise rule only fires in a money context. |
| `test_retrieve.py` | Best match comes first; no lexical overlap scores zero. |
| `test_llm.py` | Cache miss then hit, cross-instance reuse from disk, offline raises, the key depends on parameters, a corrupt cache line is skipped. |
| `test_agent.py` | Auto-handle when confident and grounded, a pre rule overrides the model, unknown intents map to `other`, fences and garbage are parsed, a blank reason still produces a reason, a string `"false"` escalate is false. |
| `test_baselines.py` | Each of the six baselines behaves as advertised. |
| `test_build_pairs.py` | Cleaning, pair linkage and `turn_index`, other brands excluded, emoji-only replies excluded, the split is by day and ordered. |
| `test_sample_golden.py` | Stratum floor and size, exclusions, first turns only, understrength strata, pool smaller than n, an impossible floor raises. |
| `test_metrics.py` | Hand-computed intent and escalation numbers. |
| `test_stats.py` | Bootstrap determinism, the paired-vs-independent distinction, McNemar against hand-computed tails, and a pin keeping the fast numpy macro-F1 equal to the sklearn one. |
| `test_judge.py` | Scores parse and clamp, a bad response sets `parse_ok=False`, and the prompt contains no system identity. |
| `test_agreement.py` | Perfect, off-by-one, reversed and zero-variance inputs. |
| `test_run.py` | The whole harness end to end on a two-row golden set with a fake transport. |
| `test_golden_integrity.py` | The golden CSVs themselves: size, unique ids, valid intents and reasons, dev disjoint from golden, no empty cells. |

### `data/` and `results/`

| Path | What it is |
|---|---|
| `data/raw/twcs.csv` | The raw Kaggle dump. Not committed. Rebuilt by `make data`. |
| `data/processed/SpotifyCares/corpus.csv` | 8000 pairs, the earlier 75% of days. The retriever's index and the baselines' training data. |
| `data/processed/SpotifyCares/holdout.csv` | 23429 pairs, strictly later in time. The sampling pool. |
| `data/golden/golden_set.csv` | 160 hand-labelled rows. The evaluation ground truth. |
| `data/golden/dev_set.csv` | 40 hand-labelled rows, disjoint by `pair_id`. The only slice tuning may touch. |
| `data/golden/labeling-guide.md` | The rules the labels were written against, fixed before labelling started. |
| `data/cache/llm_cache.jsonl` | Every LLM response keyed by its exact inputs. Committed so `make reproduce` needs no key. |
| `results/` | Written by `make eval` or `make reproduce`. Not present until one of them runs. |

### `docs/`

See `docs/00-index.md` for the index with one line on each document.

## 2. Reading order for a newcomer

Read in this order and each file only depends on ones you have already seen.

1. `src/support_agent/config.py`. Twenty lines. Tells you where everything lives and what is
   tunable.
2. `src/support_agent/intents.py`. The taxonomy is the vocabulary the rest of the code speaks in.
   Note `auto_handle_allowed`, and read `docs/06-intent-taxonomy.md` alongside it.
3. `src/support_agent/rules.py`. Short, and it is half the escalation decision.
4. `src/support_agent/retrieve.py`. Thirty lines, no surprises.
5. `src/support_agent/llm.py`. Read `chat()` first, then the cache, then the transport.
6. `src/support_agent/agent.py`. Read `SYSTEM_PROMPT`, then `handle()`. Everything above now
   makes sense as an input to those fifteen lines.
7. `src/support_agent/baselines/`. Three small files, six classes.
8. `src/support_agent/eval/metrics.py`, then `stats.py`, then `judge.py`, then `agreement.py`,
   then `run.py`.
   `run.py` last, because it is the only file that touches all the others.
9. `src/support_agent/data/build_pairs.py` and `sample_golden.py`, if you want to know where the
   CSVs came from. They run once and are not on the runtime path.

If you have ten minutes rather than an hour, read `agent.py`, `rules.py` and the summary tables in
`results/summary.md`.

## 3. How to change X

### Add an intent

1. Add an `Intent(...)` to `INTENTS` in `src/support_agent/intents.py`. Order matters: `weak_label`
   breaks ties in favour of the earlier entry, and the escalate-always intents are listed first on
   purpose. `other` must stay last.
2. Give it a definition ending in a full stop, at least two examples, an `auto_handle_allowed`
   flag, and keywords. If it is close to an existing intent, put an explicit `-> sibling`
   discriminator in the definition, as the three close pairs already do.
3. Add the intent, its two borderline rulings and its auto-handle reasoning to
   `docs/06-intent-taxonomy.md` and `data/golden/labeling-guide.md`.
4. Run the tests. `tests/test_intents.py::test_taxonomy_shape` allows 8 to 10 intents, so an
   eleventh needs that bound raised. `test_definitions_block_is_compact_and_complete` caps the
   prompt block at 1150 characters, so a long definition needs the cap raised deliberately, with a
   comment saying why.
5. The golden set is now stale: no row can carry the new label until someone re-reads the rows it
   would apply to. Say so in the report rather than silently keeping the old numbers.

### Add or change an escalation rule

1. A hard trigger on the raw message goes in the `_PRE` list in `src/support_agent/rules.py`, as a
   `(name, reason, compiled_regex)` tuple. A soft trigger on the model output goes in
   `post_check`.
2. If the rule introduces a new reason category, add it to `ESCALATION_CATEGORIES`, because
   `tests/test_golden_integrity.py` checks the golden set's `escalation_reason` column against
   that list.
3. Add a test to `tests/test_rules.py`: one message that must fire it, and one routine message
   that must not. The existing `test_routine_complaints_are_not_abusive_or_pii` exists because
   over-broad rules escalate everything and look good on recall.
4. A pre rule also changes the `RulesOnlyEscalation` baseline, since that baseline is
   `pre_check` alone. Re-run the evaluation before quoting escalation numbers.

### Swap the model

1. For one run, set `AGENT_MODEL` or `JUDGE_MODEL` in `.env`, or inline:
   `AGENT_MODEL=openai/gpt-oss-20b .venv/bin/python -m support_agent.eval.run --out results`.
2. To change the default, edit `AGENT_MODEL` or `JUDGE_MODEL` in `src/support_agent/config.py`
   and `.env.example`.
3. The cache key includes the model id, so every call is a miss on the new model. A full golden
   run costs fresh API calls and will not work offline until the new responses are cached.
4. Keep the judge in a different model family from the agent. A judge that shares a lineage with
   the thing it is grading shares its blind spots. This is recorded in decision log entry 5 and in
   `docs/04-judge-rubric.md` section 4.

### Change the thresholds

1. `CONFIDENCE_THRESHOLD` and `RETRIEVAL_THRESHOLD` live in `src/support_agent/config.py` and can
   be overridden by environment variables of the same names.
2. Do not pick them by hand. Run `.venv/bin/python -m support_agent.eval.tune`, which grid-searches
   six confidence values against six retrieval values on `data/golden/dev_set.csv` and prints the
   pair with the lowest cost-weighted error. It only ever reads the dev slice.
3. Paste the result into `config.py`, note the change in `docs/07-decision-log.md`, then re-run the
   evaluation. Tuning on golden and then reporting golden numbers is the one thing that would
   invalidate the whole result.

### Add a metric

0. If the metric is one the report will quote, give it an interval too: add it to `_stats` in
   `run.py` using `stats.bootstrap_mean_ci`, or `stats.bootstrap_ci` if it is not a mean. A point
   estimate with no interval is how the escalation claim went wrong once already (decision log 14b).
1. Add the computation to `src/support_agent/eval/metrics.py`, inside `intent_metrics` or
   `escalation_metrics`, or as a new function if it belongs to neither.
2. Add a hand-computed case to `tests/test_metrics.py`. The existing tests use small inputs whose
   answer you can work out on paper, which is the point.
3. If it should appear in the report, add a column to the relevant table in `_summary_md` in
   `src/support_agent/eval/run.py`. The dictionary written to `metrics.json` picks up new keys on
   its own.
4. Re-run `make reproduce`. Metrics are computed from `predictions.csv` and `judge_scores.csv`, so
   a new metric over existing predictions costs no API calls.

### Re-run one system only

There is no per-system flag, and there does not need to be, because the cache makes a full re-run
nearly free. Options in order of preference:

- `make reproduce` replays everything from the cache with no network and no key.
- `.venv/bin/python -m support_agent.eval.run --skip-judge --out results_tmp` runs predictions and the
  automated metrics but no judge calls, which is what you want while changing rules or thresholds.
- `.venv/bin/python -m support_agent.eval.run --limit 20 --out results` runs the first twenty
  golden rows. Note that `--limit` writes to the same `results/` directory and overwrites the full
  results, so pass a different `--out` if you want to keep them.
- To change only the judge, edit `judge.RUBRIC` and re-run. The agent's calls are cache hits and
  only the judge's calls are misses.

### Re-label a golden row

1. Edit the row in `data/golden/golden_set.csv`. Change `intent`, `should_escalate`,
   `escalation_reason` and, always, `notes`. The note must say why the previous label was wrong.
2. `should_escalate` must be `True` exactly when `escalation_reason` is not `none`, and the reason
   must be in `rules.ESCALATION_CATEGORIES`. `tests/test_golden_integrity.py` enforces both, plus
   the rule that no cell may be empty.
3. Check the decision against `data/golden/labeling-guide.md`, especially rule (b)9 on the
   reference reply and the tie-break list. If the guide does not decide the case, change the guide
   first, then apply it to every row it touches, not only the one in front of you.
4. Re-run the evaluation. Metrics change but no LLM calls are made, since predictions are keyed by
   the customer message, which did not change.
5. Record the change in `docs/03-golden-set.md`. The nine rows relabelled in the review round
   are listed there as precedent.

### Run offline versus live

Offline is the default path for anyone reviewing this repository.

```bash
make reproduce          # LLM_OFFLINE=1, replays the committed cache, no key needed
```

Live runs need a Groq key in `.env` and cost free-tier quota.

```bash
cp .env.example .env    # then paste GROQ_API_KEY
make data               # download and rebuild corpus + holdout, about 30 seconds
make golden             # re-sample the unlabelled sheets; labels are already committed
make eval               # live run, writes results/ and appends to the cache
make demo MSG="my playlist vanished after the update"
```

`LLM_OFFLINE=1` makes any prompt not already in the cache raise `CacheMissError` rather than
silently calling out, so an offline run either reproduces exactly or fails loudly.
