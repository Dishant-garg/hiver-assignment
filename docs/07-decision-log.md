# Decision log

Non-obvious decisions, in the order they were made. Each entry: what, why, what it cost.

## 1. Brand: SpotifyCares
- **What:** Build the agent for SpotifyCares only.
- **Revised from a first pass:** the first survey pass picked AmazonHelp using only a narrow
  `dm_boilerplate_share` metric (literal "DM us" phrasing), on which AmazonHelp scored lowest
  (0.6%). That metric was too narrow: it missed AmazonHelp's dominant deflection style, which is
  link-based ("please reach out to us using the link below") rather than the literal word "DM".
  A broader `deflection_any_share` metric (DM, "reach out", "contact us", "get in touch", a
  channel switch, or a bare URL) puts AmazonHelp at 44.2% deflection — not low at all. The spec's
  actual criterion is "brand replies contain real resolution steps rather than deflection," so the
  survey now also measures `resolution_step_share` (troubleshooting language: try/restart/
  reinstall/update/clear/reset/check/steps/etc.) and `customer_non_english_share` (a lexical/
  TF-IDF retriever needs mostly-English traffic to be useful). On those, AmazonHelp is worse on
  both counts than SpotifyCares.
- **Why (SpotifyCares):** 43,092 customer→brand pairs (well above the 10,000 floor, plenty for
  retrieval + a held-out golden set); 17.5% resolution-step share (replies actually walk the
  customer through a fix — restart the app, clear cache, re-login, check settings — far more than
  AmazonHelp's 10.1%); 58.1% deflection-any share, elevated but expected for an account/app-support
  brand that sometimes must move to DM for account details, and still second-lowest among the
  high-volume, high-resolution brands; only 0.8% of SpotifyCares' customer messages look
  non-English (>5 non-ASCII chars), vs. AmazonHelp's 7.9%, so SpotifyCares' traffic is close to
  wholly English and well suited to a TF-IDF/lexical retriever; median reply 131 chars, well above
  the 90-char floor.
- **Rejected:** AmazonHelp (168,814 pairs — by far the most volume — but only 10.1% resolution-step
  share and 44.2% deflection-any share once link-based hand-offs are counted, plus 7.9% of its
  customer traffic looks non-English, all of which undercut its earlier "low DM-boilerplate" case);
  AppleSupport (106,646 pairs, but 77.8% deflection-any and only 27.3% resolution-step share —
  most replies just push the customer to a DM link); Delta (42,114 pairs, only 23.9% deflection
  but also only 5.3% resolution-step share — replies mostly apologize rather than walk through a
  fix) and Tesco similarly (38,468 pairs, 6.6% resolution-step share); XboxSupport is the closest
  runner-up (23,235 pairs, 24.3% resolution-step share, 55.9% deflection) but has roughly half
  SpotifyCares' pair volume.
- **Survey table:**

| brand           |   reply_tweets |   customer_pairs |   median_reply_chars |   dm_boilerplate_share |   deflection_any_share |   resolution_step_share |   customer_non_english_share |
|:----------------|---------------:|-----------------:|---------------------:|-----------------------:|-----------------------:|-------------------------:|------------------------------:|
| AmazonHelp      |         169840 |           168814 |                  123 |                  0.006 |                  0.442 |                   0.101 |                        0.079 |
| AppleSupport    |         106860 |           106646 |                  129 |                  0.525 |                  0.778 |                   0.273 |                        0.025 |
| Uber_Support    |          56270 |            56160 |                  104 |                  0.358 |                  0.842 |                   0.059 |                        0.007 |
| SpotifyCares    |          43265 |            43092 |                  131 |                  0.308 |                  0.581 |                   0.175 |                        0.008 |
| Delta           |          42253 |            42114 |                  102 |                  0.163 |                  0.239 |                   0.053 |                        0.011 |
| Tesco           |          38573 |            38468 |                  134 |                  0.267 |                  0.321 |                   0.066 |                        0.009 |
| AmericanAir     |          36764 |            36531 |                  107 |                  0.168 |                  0.243 |                   0.086 |                        0.012 |
| TMobileHelp     |          34317 |            34215 |                  126 |                  0.818 |                  0.895 |                   0.111 |                        0.008 |
| comcastcares    |          33031 |            32921 |                  127 |                  0.713 |                  0.755 |                   0.066 |                        0.006 |
| British_Airways |          29361 |            29290 |                  124 |                  0.14  |                  0.215 |                   0.088 |                        0.008 |
| SouthwestAir    |          28977 |            28828 |                  118 |                  0.169 |                  0.293 |                   0.061 |                        0.018 |
| VirginTrains    |          27817 |            27416 |                   79 |                  0.026 |                  0.146 |                   0.043 |                        0.007 |
| Ask_Spectrum    |          25860 |            25617 |                  147 |                  0.46  |                  0.734 |                   0.092 |                        0.006 |
| XboxSupport     |          24557 |            23235 |                  115 |                  0.209 |                  0.559 |                   0.243 |                        0.004 |
| sprintcare      |          22381 |            22209 |                  117 |                  0.471 |                  0.571 |                   0.062 |                        0.014 |

## 2. Pairs, not threads
- **What:** The corpus builder (`src/support_agent/data/build_pairs.py`) reconstructs
  (customer message, brand reply) pairs, not full multi-turn threads.
- **Why:** The agent's job is to answer one incoming message; the brand's *first* reply to
  that message is the closest thing available to a ground-truth resolution, so each customer
  tweet is matched to only the first brand reply addressed to it (`in_response_to_tweet_id`
  linkage, deduplicated by customer tweet id). `turn_index` still records how deep into a
  thread the customer message sits (walked via the `in_response_to_tweet_id` parent chain),
  so multi-turn context is preserved as metadata without being required as agent input.
- **What it cost:** Real-data run on SpotifyCares (`.venv/bin/python -m
  support_agent.data.build_pairs`, ~23s): **40,167 pairs** built from the raw 2.8M-row
  `twcs.csv` (after stripping other-brand replies, URL/mention-only customer messages, and
  brand replies that clean to under `MIN_CHARS=10` — e.g. an empty/NaN reply and ~25
  emoji-only replies like "💚 /KT" — a fix-round-1 correction: both sides of a pair are now
  filtered, not just the customer side) -> **corpus 8,000** (capped by `CORPUS_MAX_PAIRS`,
  sampled with `SEED=42`) and **holdout 23,429**, of which **15,573 are first-turn**
  (`turn_index == 0`).
- **Fix round 1 (Task 4) amendment:** `clean_text` now runs `html.unescape` first, so raw
  entities (`&gt;`, `&amp;`) no longer reach prompts or retrieved precedents. Two replies that
  previously cleared `MIN_CHARS` now clean to under it (e.g. a reply whose length was mostly
  `&amp;`), so the pair total is **40,165**, not 40,167. Corpus and holdout row counts are
  unchanged (8,000 / 23,429, holdout first-turn still 15,573), but the seeded corpus sample
  redraws from a pool two rows smaller, so first-turn in corpus moves 5,001 -> 4,952.

## 3. Time split by day, 75/25
- **What:** `time_split()` splits pairs by calendar day (floor to `D`) rather than randomly:
  the earliest ~75% of days become the corpus, the remaining ~25% become the holdout, at
  `frac=0.75`.
- **Why:** A random split could put two replies from the same thread — or near-duplicate
  boilerplate from the same day — on both sides, letting a retrieval-based agent "solve" a
  holdout example by finding its own near-twin in the corpus. Splitting on calendar day makes
  the holdout strictly later in time than the corpus (`corpus.created_at.max() <
  holdout.created_at.min()`), so no holdout example's own thread, or same-day sibling, can
  ever appear in the retrieval corpus.
- **What it cost:** corpus.csv is 1.9 MB (8,000 rows, capped from a larger pre-cap corpus by
  `CORPUS_MAX_PAIRS=8000` to stay well under the 10 MB budget); holdout.csv is 5.9 MB (23,429
  rows), comfortably under 10 MB, so no additional holdout capping was needed.

## 4. Taxonomy size and auto-handle flags
- **What:** Nine intents plus `other` (`src/support_agent/intents.py`), with
  `auto_handle_allowed=False` on exactly `account_access`, `billing_and_refund`,
  `subscription_plan` and `other`, and `True` on `playback_issue`, `app_bug`,
  `library_playlist_issue`, `content_availability`, `feature_request`, `how_to_question`.
  Full derivation in `docs/06-intent-taxonomy.md`.
- **Why (size):** `scripts/explore_intents.py 12` (TF-IDF 1–2 grams + KMeans, `SEED=42`) over
  the first-turn corpus messages (4,952 after the HTML-unescape rebuild), plus 200 messages read by hand under two different
  sample seeds. The clusters do not map one-to-one onto intents: KMeans cuts on vocabulary
  ("song", "album", "playlist" spread across five clusters), while the taxonomy has to cut on
  *what is broken and who can fix it*. So the song/album/playlist clusters were split three
  ways (`content_availability` / `library_playlist_issue` / `playback_issue`) and the
  account blob three ways (`account_access` / `billing_and_refund` / `subscription_plan`),
  each split justified by the brand replying differently. Nine is the point where every
  intent still clears 3% of first-turn traffic and still has a resolution pattern no other
  intent shares; a tenth (`artist_or_royalty_issue`, `country_launch`, `ads_complaint`,
  `device_integration`) either fell under 3% or duplicated another intent's resolution.
- **Why (flags):** measured rather than assumed. For each weak-labelled intent I counted how
  often the brand's own first reply deflects out of the thread versus gives a step in-thread
  (`scripts/intent_deflection.py`, same regexes as the brand survey). The three
  escalate-always intents deflect in **74.6–88.2%** of replies; the six auto-handle
  intents deflect in **8.9–23.1%** (except `how_to_question` at 39.0%). That line coincides
  exactly with the "account access, money movement, or policy exception" rule — you cannot
  answer "why was I charged twice" or "my account was hacked" in public without the account.
- **What it cost:** `other` is **32.1%** of first-turn messages — under the 40% gate but
  large. Roughly half of it is genuinely unresolvable traffic (KMeans cluster 0, n=1543:
  praise, jokes, job pitches, partnership asks) and half is weak-labeller recall (e.g. "I got
  kicked out of my premium Spotify and it won't let me back in" is `account_access` but
  contains none of its keywords). Weak-label distribution over the 5,001 first-turn messages:
  `other` 32.1, `content_availability` 11.4, `subscription_plan` 10.7,
  `library_playlist_issue` 10.7, `billing_and_refund` 7.5, `app_bug` 7.4, `playback_issue` 7.2,
  `account_access` 6.8, `feature_request` 3.6, `how_to_question` 2.5 (%). A vote is one
  *non-overlapping matched span*, not one keyword, so nested phrases ("double charged" +
  "charged") score once; that correction alone cut `how_to_question` from 3.8% to 2.5%, which
  is the only intent below the brief's ≥ 3% rule and is kept on the strength of its distinct
  resolution rather than its volume.
  `how_to_question` is the weakest auto-handle call and the first candidate to flip to
  `False` if the eval shows it driving bad auto-handles. `definitions_block()` ships only the
  first (discriminating) example per intent to hold the per-call prompt at 1,148 chars (~255
  tokens), including an explicit `-> sibling` pointer on each of the three close pairs; the second,
  longer example stays on each `Intent` for docs and few-shot use.

## 5. Cached, rate-limited LLM client; offline reproduction mode
- **What:** `src/support_agent/llm.py` is the only module allowed to call an LLM. `LLMClient`
  wraps every call in a persistent JSONL cache (`data/cache/llm_cache.jsonl`) keyed by
  `cache_key(model, messages, temperature, json_mode, max_tokens)` (a SHA-256 of the sorted
  JSON payload, so any parameter change is a cache miss, never a stale hit — `temperature` is
  normalised via `float(...)` first so `0` and `0.0` share a key), a fixed minimum-interval
  spacer (not a token bucket: it just enforces a floor of `60/rpm` seconds, default `rpm=25`
  matched to the Groq free tier, since the previous call before allowing the next), and a
  `transport` seam so tests never touch the network — `tests/test_llm.py` passes a
  `fake_transport` and asserts cache hit/miss counts, cross-instance cache reuse from disk,
  that `offline=True` raises `CacheMissError` instead of calling out, and that a truncated
  cache line on disk is skipped (with a count in one stderr warning) rather than crashing
  `LLMClient()` at construction. The real transport (`_groq_transport`, lazy-imports
  `openai.OpenAI`/`RateLimitError` so the test suite has no hard dependency on the SDK, and
  constructs the client with `max_retries=0` so the SDK never layers its own retries under
  ours) retries 429s with backoff — preferring the `Retry-After` response header when the SDK
  exposes one, else falling back to the value parsed from Groq's `try again in Ns` message
  text — and re-raises a distinct `DailyLimitError` when the 429 text mentions a
  per-day/TPD/RPD cap, so a script that hits the daily ceiling stops cleanly and can resume
  tomorrow from the cache instead of crashing mid-run. `config.offline()` (`LLM_OFFLINE=1`)
  makes the whole pipeline replay from the committed cache with zero network calls — the
  grader's reproduction path.
- **Why:** the assignment must be reproducible by a grader who may not have (or want to spend)
  a Groq key, and the free tier's per-minute and per-day limits make a naive retry-on-fail
  loop both slow and liable to blow the daily cap mid-development. Persisting every response
  keyed by its exact inputs means re-running the pipeline (or a flaky rerun after a crash)
  costs zero additional API calls, and cache entries are already the JSON graders can open.
- **Model choice — verified 2026-09-11:** the live Groq `/models` list on that date returned 14
  ids: `allam-2-7b`, `canopylabs/orpheus-arabic-saudi`, `canopylabs/orpheus-v1-english`,
  `groq/compound`, `groq/compound-mini`, `meta-llama/llama-prompt-guard-2-22m`,
  `meta-llama/llama-prompt-guard-2-86m`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
  `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.6-27b`, `qwen/qwen3.8-27b`, `whisper-large-v3`,
  `whisper-large-v3-turbo`. **`config.AGENT_MODEL = openai/gpt-oss-120b` is served** and a smoke
  call returned `{"ok":true}`. The judge id had to change: the scaffold's placeholder
  `qwen/qwen3-32b` is **no longer served**, and neither is the brief's documented fallback
  `llama-3.3-70b-versatile` — the only text models outside the agent's own OpenAI family are the
  two Qwen 27B ids, both at ~200K tokens/day on the free tier, so the "different family, ≥ 200K
  TPD, prefer a Qwen 27B/32B id" rule lands on Qwen either way. Between the two, one live judge
  call decided it: **`qwen/qwen3.8-27b` returns a well-formed score object inside `ReplyJudge`'s
  200-token budget, while `qwen/qwen3.6-27b` spends the whole budget on reasoning tokens and the
  API rejects the truncated output with `json_validate_failed` (400).** `config.JUDGE_MODEL` and
  `.env.example` now default to `qwen/qwen3.8-27b`. The rationale for cross-family judging is
  unchanged and standard: an LLM grading its own family's lineage is likelier to share blind
  spots than to catch them.
- **What the smoke call cost:** the brief's literal smoke call
  (`max_tokens=20`, `json_mode=True`) **fails against a reasoning model** — `gpt-oss-120b` bills
  its reasoning tokens against `max_tokens`, so a 20-token ceiling truncates before any JSON is
  emitted and Groq returns `400 json_validate_failed` with an empty `failed_generation`. The same
  call at the client's default `max_tokens=400` succeeds. Nothing in the pipeline uses a ceiling
  that low (agent 400, judge 200), so this is a property of the one-off smoke probe rather than a
  live defect — but it is the first concrete confirmation that **`max_tokens` on this provider is
  a reasoning+output budget, not an output budget**, which is why the judge's 200 is the tightest
  setting in the codebase and why a more verbose reasoning model cannot be swapped in behind it
  without raising that number. The pin that follows from this is entry 8.
- **What it cost:** `_groq_transport` is untested against a live Groq error payload (`tests/test_llm.py`
  only exercises the fake transport), so the regex-based backoff/daily-cap detection is
  reviewed by reading Groq's documented 429 format rather than by a live-fire test — first
  real run should be watched for an unmatched error string.

## 6. Golden set sampling and what "should_escalate" means
- **What:** `data/golden/golden_set.csv` (160 rows) and a disjoint `dev_set.csv` (40), sampled
  by `make golden` from the **first turns of the time-held-out slice only** (15,573 messages,
  `2017-10-29` → `2017-12-03`; the training split ends `2017-10-28 23:53:56Z`), stratified by
  the keyword weak label (≥ 8 rows per stratum for golden, ≥ 2 for dev, remainder random),
  seeds 42 and 43. Every row was then hand-labelled against `data/golden/labeling-guide.md`
  with `intent`, `should_escalate`, `escalation_reason` and a written note — in four batches of
  40, reading each customer message *and* the brand's real reply. Full record, distribution
  table and the ten hardest calls: `docs/03-golden-set.md`.
- **Why the time cut matters:** the retriever is fitted on the training split, so no golden
  message and no golden reply is in the index it searches. Evaluating on a random split would
  have let the retriever find the answer to the very message being scored.
- **What `should_escalate` means:** *a correct, safe reply cannot be composed in public without
  a human* — not "the brand sent a DM". Four triggers: account access, money movement, a
  safety/legal/abuse/PII flag or an explicit request for a person, and too little information
  to answer at all. That makes the entire `account_access` / `billing_and_refund` /
  `subscription_plan` / `other` group escalate, matching `auto_handle_allowed=False` in
  `intents.py`; the six troubleshooting/information intents auto-handle unless a per-row flag
  fires. **The message decides the intent; the brand's reply decides the escalation.** Where
  the brand's first reply is a bare "DM us your username/email" with no step, link or
  explanation, the brand itself judged the message to need a human and the row escalates
  whatever the intent looks like — six rows (7, 43, 76, 108, 136, 145) turn on that rule.
  Result: **78/160 escalate (48.8%)**, six of them outside the escalate-always group.
- **What it cost:** (a) **45.6% of the weak labels were wrong** (73/160; 40% on dev), which is
  the point of the exercise but also means the set's intent distribution is
  stratify-then-relabel, **not** a prevalence estimate — `content_availability` 20.0% and
  `feature_request` 16.3% are artefacts, and `playback_issue` (4 rows), `how_to_question` (4)
  and `library_playlist_issue` (8) are too thin for stable per-intent metrics. The dev slice
  ended up with **no `app_bug` row**; tolerated because the thresholds tuned on dev are global
  rather than per-intent, with a larger dev slice as a one-more-week item. (b) Escalating whole
  intents escalates messages the brand answered in public: five plan/money rows (0, 36, 106,
  129, 151) are pure how-to questions. The eval cannot detect this (golden and agent both
  escalate, so they agree) — only a reviewer reading the escalations can, and if the verdict is
  "unnecessary" the fix belongs in the rule, not in the labels. (c) **`library_playlist_issue`
  comes out 4 escalated / 8 total** once (b)9 is applied consistently: half of an intent marked
  `auto_handle_allowed=True` needed the account. That is direct evidence for the caveat in
  `docs/06-intent-taxonomy.md` and makes it the **second candidate after `how_to_question` to flip
  to non-auto-handleable** after the eval. (d) The guide counts profanity aimed at the product
  as frustration, so **no row is labelled `abusive`** while `rules.pre_check` fires `abusive` on
  golden 70, golden 144 and dev 37 (and `safety_legal` on dev 20). The disagreement is over the
  **reason category only** — all four rows are already `should_escalate=True` in the golden set,
  so the eval sees reason-level mismatches, not false escalations. (e) Single labeller,
  AI-assisted, no inter-annotator agreement: the labels were reviewed before any number computed
  from them was quoted, and every row carries the reasoning needed to challenge it. That review
  round after the first pass
  relabelled nine rows (16, 41, 43, 44, 62, 108, 136, 139, 145), almost all because (b)9 had
  been overridden by the labeller's own view of what the published answer should have been; the
  guide's rules were tightened so the rule, not the judgement call, carries the decision.

## 7. Escalation thresholds: 0.0 confidence, 0.3 retrieval

- **What:** `.venv/bin/python -m support_agent.eval.tune` grid-searched
  `conf_threshold ∈ {0.0, 0.5, 0.6, 0.7, 0.8, 0.9} × retr_threshold ∈ {0.0, 0.05, 0.1, 0.15, 0.2, 0.3}`
  on the 40-row dev slice (one agent call per message, thresholds re-applied post-hoc through
  `rules.post_check`, so the grid costs 40 calls, not 1,440). Winner:
  `{"conf_threshold": 0.0, "retr_threshold": 0.3, "weighted_error": 0.25, "recall": 0.963, "automation_rate": 0.225}`.
  `config.CONFIDENCE_THRESHOLD` and `config.RETRIEVAL_THRESHOLD` now default to those.
- **The grid has only three distinct outcomes, and the confidence axis is not one of them.** All six
  confidence values produce byte-identical metrics at every retrieval value:

  | retr | weighted error | recall | precision | automation rate |
  |---|---|---|---|---|
  | 0.0 – 0.15 | 0.275 | 0.926 | 0.962 | 0.350 |
  | 0.20 | 0.300 | 0.926 | 0.926 | 0.325 |
  | 0.30 | **0.250** | **0.963** | 0.839 | 0.225 |

  So `conf_threshold=0.0` was not selected on merit — it is the first of six tied candidates under
  the tie-break (lowest `weighted_error`, then highest `automation_rate`, both identical across the
  axis). **The confidence gate is dead code on this slice.** The agent is confident almost
  everywhere (median 0.97, min 0.45, 25th percentile 0.9375), and the handful of genuinely
  low-confidence rows are already escalating for an independent reason — a `pre_check` rule hit, a
  non-auto-handleable intent, or the LLM's own `escalate` flag — so raising the gate to 0.9 changes
  no decision. Shipping 0.0 says that plainly; shipping the old untuned 0.6 would have implied a
  gate that was doing work.
- **The trade-off actually bought at `retr=0.3`:** one more true escalation caught
  (recall 0.926 → 0.963, i.e. 25 → 26 of 27) in exchange for four more unnecessary ones
  (precision 0.962 → 0.839) and a third of the remaining automation (automation rate 0.350 →
  0.225). `weighted_error` prefers this because it prices a missed escalation above an unnecessary
  one, which is the right asymmetry for support — a wrongly-escalated ticket costs an agent a
  minute, a wrongly-automated one costs a customer their account. It is still worth stating in
  plain terms: **the tuned configuration hands 77.5% of dev messages to a human.** That is a
  defensible operating point for a first deployment, not an impressive one.
- **What it cost:** (a) **`retr=0.3` is the top of the grid**, so the search stopped at its own
  boundary and the true optimum may lie above it — the top retrieval score has median 0.342 and
  75th percentile 0.397, so a threshold of 0.4+ is on the table and would escalate still more.
  Extending the grid upward is a one-line change and 0 extra API calls (the dev decisions are
  cached); it was left alone because pushing automation below 22.5% to chase a 40-row metric is
  not a decision this slice can support. (b) **40 rows, 27 of them escalations.** Each dev row is
  worth 2.5 points of recall, so the 0.275 → 0.250 improvement that picked the winner is *one
  row*. Treat the chosen point as "retrieval gate on, confidence gate off", not as two calibrated
  numbers. (c) **The dev slice contains no `app_bug` row at all** (its eight intents are
  `account_access` 8, `billing_and_refund` 7, `other` 6, `subscription_plan` 5, `feature_request` 4,
  `content_availability` 4, `playback_issue` 3, `library_playlist_issue` 3), so the thresholds were
  never exercised against the one intent whose escalation behaviour is least like the others. The
  thresholds are global rather than per-intent, which limits the damage, but a larger dev slice
  remains the top of the one-more-week list (see entry 6).

## 8. Reasoning effort pinned to "low" for gpt-oss models

- **What:** `src/support_agent/llm.py` defines `GPT_OSS_PREFIX = "openai/gpt-oss"` and
  `GPT_OSS_REASONING_EFFORT = "low"`, and `_groq_transport` sends
  `reasoning_effort="low"` on every request whose model id starts with that prefix. Nothing else
  changed: `max_tokens` stays at the client default of 400, `agent.py` is untouched, and the Qwen
  judge never receives the parameter.
- **Why — this was forced, not chosen.** The first live `eval.tune` run died three rows into the
  dev slice with `400 json_validate_failed` and
  `failed_generation: "max completion tokens reached before generating a valid document"`.
  Per entry 5, `max_tokens` on Groq is a **reasoning + output** budget, so `gpt-oss-120b`'s
  reasoning tokens are billed against the agent's 400-token ceiling. Measuring completion tokens
  on four dev messages at both settings:

  | dev row | prompt | completion @ default effort | completion @ low |
  |---|---|---|---|
  | 0 | 791 | **405** | 162 |
  | 1 | 764 | 245 | 113 |
  | 2 | 726 | 215 | 129 |
  | 3 | 752 | 348 | 178 |

  Row 0 needs 405 against a 400 ceiling. **Roughly a quarter of rows truncate**, at an arbitrary
  point in a 160-row run — the two Task 8 demos succeeded earlier by luck, not by margin. At
  `"low"` the same turns cost 113-178 completion tokens, comfortably inside the ceiling.
- **Why the transport rather than `agent.py`:** three fixes were possible and only this one
  satisfies every constraint. (a) Raising `max_tokens` in `agent.py` is out — that module is frozen
  for this task. (b) Raising `llm.chat`'s default to ~1024 changes `cache_key` for **every** call,
  breaks `tests/test_llm.py`'s `cache_key("m", msgs, 0.0, True, 400)` which is written against that
  default, and blows the daily budget: at default effort the pipeline averages ~1,060 tokens/call
  and 200 calls (40 tune + 160 golden) is ~212K against a ~200K/day cap. (c) Pinning the effort
  fixes the actual problem — the model was spending more on reasoning than an intent-plus-draft
  task needs — and drops the same 200 calls to ~180K, inside the cap. The prefix guard keeps it
  off the Qwen judge, which does not take the parameter in this form.
- **Deliberately outside `cache_key`.** `reasoning_effort` is a transport constant, with the same
  standing as the `max_retries=0` and `timeout=60` already set there, and it is **not** a
  `cache_key` input. That is why it is a module constant rather than an `os.getenv` override: if it
  could vary per run, a cached response generated under one reasoning budget could silently answer
  a prompt issued under another — exactly the stale-hit failure entry 5 claims the cache design
  prevents. A constant cannot drift between runs, so the invariant holds. As a tripwire against a
  future edit, `LLMClient.chat` writes the effort into the cache line's human-readable `tag`
  (`"agent@low"`); `_load` reads only `key` and `response`, so this cannot affect cache hits, but a
  change to the effort now shows up as a diff in the committed cache instead of passing unnoticed.
- **The cache was cleared and regenerated after the change.** Seven lines had been produced under
  the old default effort (one smoke call, one judge probe, two demos, three tune rows); they were
  backed up outside the repo and the file was emptied before any recorded run. Every committed line
  was therefore generated under the configuration that ships: **202 agent lines** (2 demo + 40 tune
  + 160 golden) plus 360 judge and 40 judge-at-0.7, 602 in total.
- **What it cost:** (a) **Lower reasoning effort is a quality choice made under a budget
  constraint, not a free win.** The agent's golden-set numbers (intent 0.744, judged reply overall
  3.07) are all measured at `"low"`; nobody has measured what the same pipeline scores at default
  effort, because doing so costs ~212K tokens against a ~200K/day cap — two days of budget for one
  comparison. If the free-tier cap were lifted, re-running the golden set at default effort is the
  first experiment worth doing, and it is plausible that some of the groundedness weakness
  (3.24, the agent's worst judged dimension) is reasoning the model was not given room to do.
  (b) `_groq_transport` remains the one module the suite cannot exercise end to end against a real
  Groq payload; `tests/test_llm.py` now asserts the parameter is sent for gpt-oss and withheld for
  Qwen through a stubbed `openai.OpenAI`, which covers the branch but not the provider's response
  to it. (c) The prefix test is a string match, so a future Groq id that reasons the same way under
  a different name would not be covered.

## 9. Judge agreement study design

**Decision.** Validate the LLM judge against hand-applied rubric labels on **60 replies** --
30 agent, 20 nn, 10 canned -- scored **blind** (no judge score, rationale or sub-score visible, and
the `system` column masked while scoring), then report quadratic-weighted kappa, Spearman, exact and
within-one agreement overall and per system, alongside the judge's temperature self-consistency.
Results: `docs/05-judge-agreement.md`, `results/judge_agreement.json`.

- **Why 60.** The study exists to answer one question -- can the judge's `overall` be used as a
  cheap proxy for the golden evaluation -- and the answer is an aggregate agreement statistic, not a
  per-row verdict. Sixty rows is where hand-scoring stops being cheap (each row means reading a
  customer message, a reference reply and a candidate, and writing a defensible note; 60 is a couple
  of hours of careful work) and is still enough that the headline kappa is not noise. Scoring all
  160 golden rows x 3 systems would have been 480 labels for a number that would not change the
  verdict.
- **Why 30/20/10 rather than 20/20/20.** The agent is the system under test: its slice carries the
  claim the report actually makes, so it gets the tightest error bar. nn is the baseline the agent
  must beat, and it is the slice where the judge's evidence-anchoring bias should show up most
  (an nn reply often *is* the retrieved evidence), so it gets the second-largest share. `canned` is a
  single constant string -- one candidate text, 160 identical replies -- so extra rows there buy
  almost nothing: they vary only in which customer message the same sentence is answering. Ten is
  enough to confirm both scorers punish it, which is all the control is for. The cost of this
  choice is that per-system kappa on 10 rows is uninterpretable, and the document says so rather
  than quoting it as a finding.
- **Why blind, and what blinding cost.** An unblinded scorer anchors on the judge's number, which
  would turn the study into a measure of the scorer's deference rather than of the judge. The
  sampler therefore writes a sheet with no judge output in it at all, and the judge file was not
  opened until every label was on disk. One hole had to be patched honestly: the judge run covers
  canned on `golden_id` 0-39 only (budget), so 8 canned rows in the first draw had no counterpart
  and were redrawn *after* that was discovered -- those 8 were scored knowing they were canned. The
  50 agent/nn rows were untouched by this. The sampler now reads the judge file's
  `(golden_id, system)` key columns via `usecols` so the constraint is enforced up front without a
  score ever reaching the sheet.
- **What the numbers licence.** kappa 0.539 (95% CI [0.331, 0.697]) / Spearman 0.566 ([0.336, 0.740])
  / within-one 0.78 on n=58, with judge and human means 0.02 apart and both ranking the agent first,
  licences **"the judge can rank systems"** -- specifically, that the agent beats the baselines. The
  intervals are seeded percentile bootstraps (`config.SEED`, 2,000 resamples) added in
  `scripts/judge_agreement.py`; they exclude zero, so the agreement is real, but 0.539 is only
  **moderate** on Landis-Koch (substantial starts at 0.61) and the interval is wide enough that no
  claim resting on a kappa threshold is supportable. It does **not** licence ranking nn against
  canned -- the two scorers order them differently on gaps of 0.10 and 0.35 points, and the canned
  slice is confined to golden ids 0-39, which are not representative of the golden set (35% vs 15%
  content_availability, no playback_issue or how_to_question), so its mean describes
  canned-on-those-rows rather than canned overall. With exact agreement at 0.38 and 13 of 60 rows
  differing by >= 2 points it does **not** licence citing a judge score for an individual reply or
  using the judge as a production gate. The self-consistency ceiling of 0.704 was measured on 40
  **agent** replies, so the like-for-like comparison is the agent-slice kappa 0.571 ([0.339, 0.745]),
  whose interval overlaps it heavily.
- **What it does not settle.** The scorer was an AI assistant applying the written rubric, not a
  second human, so there is no human-human baseline and no way to rule out correlated error between
  scorer and judge -- both are LLMs and can be wrong in the same direction, which this study would
  record as agreement. The largest disagreements are systematic rather than random: rationales that
  say the candidate "invents" something average -1.06 against the human, rationales praising an exact
  match to the evidence average +1.50, i.e. the judge rewards fidelity to retrieved text over whether
  the customer is helped. That bias runs *against* the agent and *for* the retrieval baseline, so the
  agent's margin is, if anything, understated.

## 10. One LLM call, not three

- **What:** `agent.handle` makes a single structured call that returns intent, confidence, draft
  reply, escalate flag and reason in one JSON object. Not three calls (classify, then draft, then
  decide).
- **Why:** All three tasks read the same context -- the customer message, the intent definitions and
  the three retrieved exchanges. Splitting them means sending that context three times. A full
  golden run is 160 agent calls; at three calls each it is 480, plus 360 judge calls, against a free
  tier capping around 200K tokens per day per model. The evaluation would have taken three sittings
  instead of one and the reproduction cache would be three times the size for no new information.
  Drafting also genuinely benefits from knowing the intent, and the model is better placed to keep
  those consistent inside one completion than a pipeline is at stitching them together.
- **What it cost:** Two things, and the second shows up in the results. The tasks cannot be ablated
  independently -- there is no way to ask "how good is the reply given a correct intent?" without a
  second run. And an intent error becomes a routing error with no intervening check, which is
  failure mode 2 in `docs/01-report.md`: four of the nine missed escalations (golden 28, 73, 139, 142)
  are rows where the model picked an auto-handleable intent and the gate then let its own draft
  through. A separate classifier would not have been more accurate, but it would have made the
  failure visible as a classification error rather than a safety one.

## 11. TF-IDF retrieval, not embeddings

- **What:** `retrieve.py` is scikit-learn `TfidfVectorizer` plus cosine similarity over the 8,000
  corpus pairs, top k = 3. No embedding model, no vector store.
- **Why:** Groq serves no embedding endpoint, so embeddings would mean either a second provider (a
  second key for a grader to obtain) or a local sentence-transformer, which drags in torch and turns
  `make setup` from a 30-second pip install into a multi-gigabyte download. The reproduction budget
  is 15 minutes on a laptop with no key. TF-IDF fits in about a second and the whole index is a
  pickle-free rebuild from a committed CSV.
- **What it cost:** More than expected, and the report says so. The median top retrieval score over
  the 160 golden rows is 0.319 against a 0.3 threshold, so the scores are compressed right where the
  gate sits. On 60-character tweets, lexical overlap finds the brand's *tone* and not its *content*:
  for golden 7 the three retrieved replies are "What's happening exactly?", "Can you try the steps
  here" and a question about 3G versus WiFi, none of which contain a product fact. That is the
  proximate cause of failure mode 1 (the agent inventing UI paths, golden 28, 73, 142) and of
  failure mode 4 (32 of 39 unnecessary escalations fire on `weak_evidence` alone). Embeddings would
  likely not have fixed it either, because the corpus itself is mostly deflection; the real fix is a
  different corpus (help-centre articles), which is item 3 on the one-more-week list.

## 12. The rule sandwich: deterministic checks on both sides of the model

- **What:** `rules.pre_check` runs on the raw message before the model sees it
  (`human_requested`, `safety_legal`, `pii`, `abusive`). `rules.post_check` runs on the parsed
  output (`low_confidence`, `weak_evidence`, `intent_not_auto_handleable`, `forbidden_promise`,
  `too_long`). Either side firing forces an escalation, and the reason string is the rule names
  joined, not model prose.
- **Why:** A model's opinion about its own confidence is not evidence. The escalation decision is
  the part of this system that can hurt someone, so it has to be readable, testable and changeable
  by a person who does not want to re-run an evaluation. Rules are all three. As a side effect the
  same layer is the "simple" escalation baseline, which costs nothing extra to report.
- **Revised after an audit of the golden set.** Auditing the 160 labels against the labelling guide
  turned up no labelling error, but it did turn up three places where `pre_check` **contradicted
  the guide it was supposed to implement**:
  - `abusive` matched bare profanity, so it flagged *"Your web player fucking blows"* — which is
    the guide's own counter-example for rule (b2): "frustrated swearing *about* the product is not
    abuse on its own; abuse aimed at a person is". Profanity now only counts when pointed at a
    person.
  - `human_requested` implemented only half of rule (b4). The guide counts asking "to be
    phoned/DM'd by a human", and a bare *"please see my DM"* is the commonest way this brand's
    customers ask for a person — the rule saw none of the five such rows in the golden set.
  - `safety_legal` missed the phishing and discrimination language rule (b1) names.
  The patterns were rewritten from the guide's wording, not from the golden outcomes; the guide was
  fixed before labelling, so this is an implementation catching up with a pre-registered spec
  rather than tuning on the evaluation set.
- **What the correction did, including the part that hurts.** `rules_only` recall went 0.026 →
  0.103 at unchanged precision 1.000, and weighted error 2.375 → 2.188. More to the point, the
  rule's *reason* now matches the hand-recorded reason on 8 of 8 rows where it fires; before, it
  matched on **0 of 2**. The old `rules_only` precision of 1.000 was an artefact: it fired on
  exactly two rows, both false "abusive" matches that happened to land on rows escalating for
  unrelated reasons. Right answer, wrong reason, twice.
  Against that, the agent got slightly **worse** — weighted error 0.525 → 0.556, misses 9 → 10 —
  because golden 70 (*"Your web player fucking blows"*, hand-labelled `other` / `ambiguous`) had
  been escalating *only* on the bad abuse match. With the bug gone the agent classifies it
  `app_bug`, which is auto-handleable, and auto-handles it. A bug was masking a real classification
  weakness, and fixing the bug exposed it. The regression is well inside the [-0.156, +0.263] band
  on that comparison, and keeping a defect because it accidentally helps is not a decision worth
  defending.
- **What it cost:** Both directions are visible in `results/metrics.json`. Alone the rules are still
  not a safety mechanism: `rules_only` catches 8 of 78 true escalations, weighted error 2.188. And
  in the full system the post-check rules are what caps automation at 33.1%, since all 39
  unnecessary escalations are rule hits (32 `weak_evidence`, 4 `intent_not_auto_handleable`, 2 both,
  1 `forbidden_promise`) rather than the model asking for help. The model's own escalate flag fires
  on only 20 of the 107 escalations. The rules are doing the work, which is what was wanted, but it
  means the system's escalation behaviour is mostly a property of two thresholds and one boolean
  column in the taxonomy, not of the model.

## 13. `other` never auto-handles, and a bare "DM us" reply counts as an escalation

- **What:** Two boundary choices that widen escalation past what the brand itself does.
  `other` carries `auto_handle_allowed=False` in `intents.py`, alongside the three account/money
  intents. And rule (b)9 of `data/golden/labeling-guide.md` labels a row `should_escalate=True` when
  the brand's real reply is a bare request to move to DM with no step, link or explanation in it.
- **Why:** `other` is the bucket for "the taxonomy does not cover this", which includes artist and
  royalty matters, creator-platform questions and anything unparseable. A reply the agent is not
  confident enough to classify is not a reply it should send. The DM rule exists because the golden
  set has to encode what *should* happen, not what the brand did: when SpotifyCares answers a lost
  library with "DM us your username", a human is taking the case over, and an evaluation that scored
  that as automation would reward the agent for producing the same deflection.
- **What it cost:** Both sides of the ledger, quantified in `results/failures.md`. The DM rule
  flipped six otherwise auto-handleable rows (7, 43, 76, 108, 136, 145) and pushed the golden
  escalation rate to 48.8%, above the 45.0% share of the escalate-always intents. Five of those six
  are now missed escalations (all but 43), including golden 145, where the agent wrote "Please DM us
  your account email" and set `escalate=False` -- the judge scored that reply 5 on every dimension. Meanwhile
  `other` produces unnecessary escalations in the opposite direction: golden 1, 11, 18 and 116 were
  all filed `other` and escalated, and the judge gave golden 1 a 5. One label is carrying both "I do
  not know" and "a human must look at this", which is why splitting it is item 4 on the
  one-more-week list.

## 14. Two decisions about not overstating the result

Both halves of this entry are the same decision applied twice: where the evaluation could have
been made to look better, report the weaker version and say why.

### (a) The simple baselines train on weak labels, not golden labels

- **What:** `baselines/intent.py` fits TF-IDF plus logistic regression on the keyword weak labels
  over the corpus. It never sees a hand label. `baselines/reply.py` returns the nearest neighbour's
  historical reply verbatim, with no rewriting.
- **Why:** The golden set is 160 rows and it is the only ground truth in the project. Training any
  baseline on it, even under cross-validation, means the comparison is no longer a held-out one, and
  the 40-row dev slice is too small to fit a ten-class classifier on. Weak labels are free and
  available for all 8,000 corpus rows, so the baseline is trained on data the agent also never used
  for anything but sampling.
- **What it cost:** The headline intent comparison is against a handicapped opponent, and
  `docs/01-report.md` section 4 says so rather than banking the margin. The weak labeller disagrees
  with the hand labels on 45.6% of golden rows, so logreg is being asked to reproduce a signal that
  is itself about 46% noise; its 0.463 accuracy understates what a TF-IDF classifier could do with
  clean labels. The agent's 0.738 is still clearly ahead, but "27 points above logistic regression"
  is a softer claim than it sounds. The fix is a nested cross-validation on the 160 golden rows,
  reported alongside rather than instead of this number, and it costs no API calls.

### (b) Report intervals, not point estimates — and let one overturn a claim

- **What:** Every headline metric ships with a seeded bootstrap 95% CI, and every system
  comparison is run **paired** — both systems scored on the same rows, bootstrapping the gap —
  in `src/support_agent/eval/stats.py`, surfaced in `results/summary.md` and `metrics.json`.
  Intent adds an exact McNemar test on the discordant rows.
- **Why:** the evaluation's central comparison (agent vs always-escalate on cost-weighted error)
  turned on a difference of 0.0125. Reporting that as a number invites a conclusion, and an
  earlier draft of the report duly drew one: "the agent loses this table." The paired bootstrap
  says the gap is +0.0125 **[-0.181, +0.225]**, with the agent cheaper in 46% of resamples — an
  interval eighteen times wider than the difference inside it. The claim was not supportable in
  either direction. A take-home that asks you to prove a system is trustworthy is partly asking
  whether you can tell a result from a rounding error.
- **Why paired specifically:** the systems are scored on identical rows and fail on largely the
  same messages. Comparing two independent per-system intervals would have hidden a real result:
  the agent's reply-quality lead over nearest-neighbour is +0.400 **[+0.106, +0.688]** paired —
  excludes zero, ahead in 100% of resamples — while the two systems' own intervals ([2.819, 3.306]
  and [2.406, 2.925]) overlap. Unpaired, that finding disappears.
- **What it changed:** three claims. The escalation comparison went from "loses" to
  "indistinguishable". The reply-quality lead went from hedged ("sits inside the noise") to
  supported. The intent result was confirmed and strengthened (p = 2.6e-07).
- **Cost:** ~1s of runtime and a duplicated macro-F1 implementation (`_macro_f1_from_codes`, in
  numpy, because routing 10,000 resamples through sklearn dominated everything else). A test pins
  the duplicate against `intent_metrics` so it cannot drift.
- **The limit, stated because an interval is easy to over-read:** these CIs cover *sampling noise
  over golden rows only*. They say nothing about labelling error, the stratify-then-relabel
  distribution, the choice of `miss_cost`, one brand, or one model at one reasoning effort —
  which are the larger error terms and have no interval at all. Report section 7 says this too,
  in the section where it does the most damage to my own headline.

## 15. Temperature 0 everywhere, and the cache is committed

- **What:** Every call in the pipeline runs at temperature 0 -- the agent, the judge, and the
  threshold tuning pass. The one exception is deliberate: `scripts/judge_consistency.py` re-scores
  40 agent replies at 0.7 specifically to measure how much the judge moves. All 602 responses are
  committed to `data/cache/llm_cache.jsonl`, keyed by a SHA-256 of the exact inputs.
- **Why:** The cache is the reproduction story. `make reproduce` sets `LLM_OFFLINE=1`, which turns
  any prompt not already on disk into a `CacheMissError` rather than a network call, so a grader
  with no API key gets either the exact published numbers or a loud failure. That only means
  something if the model was asked to be deterministic in the first place; a cache of sampled
  responses is a record of one lucky draw dressed up as a result. Temperature 0 also removes one
  free parameter from every comparison in the report, so a difference between the agent and a
  baseline cannot be sampling noise in the agent.
- **What it cost:** There is no variance estimate on the agent's own output. Every agent number in
  `results/` is a single sample, so the error bars in the report come from the 160-row sample size
  and from the judge study, never from re-running the model. The only measured variance in the whole
  project is the judge's temperature arm (kappa 0.704 self-agreement), and it covers the judge, not
  the agent. Committing the cache also means the repository ships 602 responses from one model
  version on one date: it proves the pipeline is deterministic and proves nothing about
  `gpt-oss-120b` next month. Re-running live is the only way to find out, and it costs a day of free
  tier quota.
