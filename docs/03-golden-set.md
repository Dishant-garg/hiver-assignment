# Golden set — sampling, labelling, and what the labels mean

`data/golden/golden_set.csv` (160 rows) is the evaluation ground truth for this project.
`data/golden/dev_set.csv` (40 rows, disjoint by `pair_id`) is the only slice that thresholds
and prompts may be tuned on. The rules applied are in `data/golden/labeling-guide.md`; this
document records how the rows were chosen, how they were labelled, what came out, and what to
distrust.

> **Provenance, stated plainly.** These labels were produced with AI assistance: a single
> model-driven labelling pass, applying a guide written before labelling started, with a
> written justification on every row. They **were then reviewed**: a review round after the first
> pass relabelled nine rows (16, 41, 43, 44, 62, 108, 136, 139, 145) before any number computed
> from them was quoted as a result — decision log entry 6(e). No second annotator has seen them,
> so there is no inter-annotator agreement figure — see *Limitations*.

## 1. Sampling

Produced by `src/support_agent/data/sample_golden.py` (`make golden`), reproducible from the
processed data alone.

| property | value |
|---|---|
| pool | `data/processed/SpotifyCares/holdout.csv`, first turns only (`turn_index == 0`) — **15,573 messages** |
| time cut | the holdout is the later 25% of days: train ends `2017-10-28 23:53:56Z`, holdout runs `2017-10-29 00:00:03Z` → `2017-12-03 22:07:39Z`. No message in either labelled set was available to the retriever's index. |
| stratification | by `intents.weak_label` (the keyword vote), ≥ 8 rows per stratum for golden and ≥ 2 for dev; the remainder filled at random from the same pool |
| seed | `config.SEED = 42` for golden, `43` for dev; dev additionally excludes every golden `pair_id` |
| sizes | golden 160 (`config.GOLDEN_SIZE`), dev 40 (`config.DEV_SIZE`) |

Stratifying on the weak label is what keeps rare intents (`feature_request`,
`how_to_question`) in the set at all — a uniform sample of 160 would have been dominated by
`other` and the catalogue questions. It also introduces a bias, recorded below.

## 2. Labelling procedure

1. **Guide first.** `data/golden/labeling-guide.md` was written before any row was labelled:
   intent table with two borderline rulings per intent, the escalation rules, nine tie-break
   rules, and the standing instruction that the label describes the *customer's message*, not
   the brand's reply.
2. **Four batches of 40** (plus the 40-row dev slice as a fifth). Each batch was printed as
   `golden_id, customer_text, reference_reply, weak_label` and read in full — every message
   **and** its reference reply — before any label was written.
3. **Labels appended to a working CSV** (`labels_working.csv` / `labels_working_dev.csv`,
   deleted after the merge) carrying `golden_id, intent, should_escalate, escalation_reason,
   notes`. A note was written on **every one of the 200 rows**, not only the hard ones; rows
   where the weak label was rejected say so explicitly.
4. **Retro-fixes.** Row 32 was first labelled `ambiguous` and changed to `human_requested` when
   row 114 turned out to be the same situation ("answer my DM"). A review round after the first
   pass then relabelled nine more rows — 16, 41, 43, 44, 62, 108, 136, 139, 145 — almost all of
   them for one reason: the reference-reply rule (b)9 had been applied to some rows and
   overridden on others by the labeller's own view of what the published answer should have
   been. The guide's rules (b)9, tie-break 9 (the surface test) and tie-break 10 (payment
   methods) were tightened at the same time so the rule, not the judgement call, carries the
   decision. Old → new for every relabelled row is listed in the task report.
5. **Merge** with the snippet in the task brief, asserting that every intent is in
   `intents.INTENT_NAMES`, every reason is in `rules.ESCALATION_CATEGORIES`, and
   `should_escalate == (escalation_reason != "none")`. The unlabelled inputs and both working
   files were then deleted; they are reproducible with `make golden`.
6. **`tests/test_golden_integrity.py`** re-checks the shape and the label contract on every
   test run, so a hand edit to the CSV cannot silently break the evaluation.

### What `should_escalate` means here

It does **not** mean "the brand sent a DM". It means: *a correct, safe reply to this message
cannot be composed in public without a human*. Four things put a row over that line —
account access, money movement, a safety/legal/abuse/PII flag or an explicit request for a
person, and not enough information to answer at all. Concretely this makes the whole
`account_access` / `billing_and_refund` / `subscription_plan` / `other` group escalate, which
matches `auto_handle_allowed=False` in `intents.py`, and leaves the six troubleshooting and
information intents auto-handleable unless a per-row flag fires — which it does for six of
them (rows 7, 43, 76, 108, 136, 145), all under the reference-reply rule.

## 3. Label distribution

**Golden (160).** `weak n` is how many rows the keyword labeller assigned to that intent;
`agree` is how many of the final rows the weak labeller got right.

| intent | n | share | escalated | weak n | agree |
|---|---:|---:|---:|---:|---:|
| `account_access` | 13 | 8.1% | 13 | 13 | 9 |
| `billing_and_refund` | 20 | 12.5% | 20 | 15 | 14 |
| `subscription_plan` | 19 | 11.9% | 19 | 17 | 14 |
| `playback_issue` | 4 | 2.5% | 0 | 11 | 3 |
| `app_bug` | 14 | 8.8% | 1 | 14 | 6 |
| `library_playlist_issue` | 8 | 5.0% | **4** | 12 | 5 |
| `content_availability` | 32 | 20.0% | 0 | 19 | 14 |
| `feature_request` | 26 | 16.3% | 0 | 9 | 7 |
| `how_to_question` | 4 | 2.5% | 1 | 9 | 2 |
| `other` | 20 | 12.5% | 20 | 41 | 13 |
| **total** | **160** | | **78 (48.8%)** | | **87 (54.4%)** |

Escalation reasons, golden: `none` 82, `needs_account_access` 36, `billing_dispute` 16,
`out_of_scope` 10, `human_requested` 8, `ambiguous` 5, `safety_legal` 3. No row was labelled
`abusive` or `pii` — see hardest call #2.

**`library_playlist_issue` is split 4 escalated / 8 total.** Those four (rows 43, 108, 136,
145) are the rows where the brand's reply is a bare "DM us your username or email" with no
step, link or explanation in it, so guide rule (b)9 escalates them; the other four got a
published fix in-thread (the "Downloads unexpectedly removed" article, the suggested-tracks
explanation). A 50% escalation rate on an intent marked `auto_handle_allowed=True` is direct
evidence for the caveat in `docs/06-intent-taxonomy.md` — that genuinely lost libraries still get
pulled into DM — and makes `library_playlist_issue` the **second candidate, after
`how_to_question`, to flip to non-auto-handleable** once the eval has numbers.

**Dev (40).** `billing_and_refund` 8, `account_access` 7, `other` 7, `subscription_plan` 5,
`feature_request` 4, `content_availability` 4, `playback_issue` 2, `library_playlist_issue` 2,
`how_to_question` 1, **`app_bug` 0**. Escalated: 27 (67.5%). Reasons: `none` 13,
`needs_account_access` 12, `billing_dispute` 8, `out_of_scope` 5, `human_requested` 1,
`ambiguous` 1.

**Weak-label disagreement: 45.6% on golden (73/160) and 40.0% on dev (16/40).** That is the
headline number for why the weak labeller is a baseline and not ground truth. The two biggest
movements are the ones `docs/06-intent-taxonomy.md` predicted:

- **`other` collapses, 41 → 20.** Half of the keyword labeller's `other` rows carry
  a perfectly legible intent (catalogue asks with no keyword: "why is acid rap not on ???",
  "get more ben fankhauser stuff"; country launches; feature asks phrased as complaints).
- **`feature_request` almost triples, 9 → 26, and `content_availability` grows 19 → 32.** Both are
  intents the keyword lists under-detect, and both absorb rows the labeller had filed as
  `other`, `app_bug` or `playback_issue`.
- Against that, **`playback_issue` shrank 11 → 4**: most of its keyword hits ("streaming",
  "volume", "queue", "skip") appear in messages that are really feature requests or catalogue
  questions. It is the smallest non-`other` class in the golden set at four rows, which is too
  few to measure per-intent accuracy on with any confidence.

Escalation rate (48.8% golden) sits above the 45.0% share of the escalate-always intents
(`account_access` + `billing_and_refund` + `subscription_plan` + `other` = 72 rows), because
six rows escalate outside that group — **7, 43, 76, 108, 136, 145**, hardest calls #1 and #10
below and the four `library_playlist_issue` rows — and no row inside it was let through.

## 4. The ten hardest calls

1. **Row 7 — "Why does Spotify not sync to Facebook anymore?"** → `app_bug`, escalate
   `needs_account_access`. An integration failure rather than a retired feature (they ask why
   it broke, not for something new), but the brand did not understand it either and pulled it
   into DM with nothing else in the reply — guide rule (b)9. Together with row 76 and the four
   `library_playlist_issue` rows (43, 108, 136, 145), this is one of **six** rows where the
   reference-reply rule escalated an otherwise auto-handleable intent.
2. **Row 70 "your web player fucking blows", row 144 "WTF", dev row 37 (an explicit track
   title)** → **not** `abusive`. The guide counts profanity aimed at the *product*, or merely
   quoted, as frustration and reserves `abusive` for attacks on a person; nothing in either set
   cleared that bar, so the set contains **zero `abusive` rows** while `rules.pre_check` fires
   `abusive` on those three (and `safety_legal` on dev row 20, where "fraud" describes the
   customer's own bank incident). **The disagreement is over the reason category only, not over
   whether to escalate:** every row where `pre_check` fires is already `should_escalate=True`
   in the golden set, labelled `ambiguous`, `needs_account_access`, `out_of_scope` and
   `billing_dispute` respectively. So the eval will show reason-level mismatches, not false
   escalations, and the right place to argue about the threshold is the rule, not the ground
   truth.
3. **Row 103 — "someone hacked my account and I've lost my playlists, can I retrieve them?"**
   → `account_access`, escalate. Two intents in one sentence; tie-break 1 gives it to the
   escalate-always side. The brand answered it with a self-serve link, and I still escalated:
   no agent should promise playlist recovery on a compromised account it cannot see. Row 159
   and dev row 16 got the same treatment.
4. **Row 106 — "Do I need to cancel and resubscribe to move to the annual plan?"** →
   `subscription_plan`, escalate. A pure how-to that the brand answered fully in public. It is
   escalated anyway because the whole money/plan class is, and `docs/06-intent-taxonomy.md` names
   this exact case as the accepted cost of that boundary. Rows 0, 36, 129 and 151 are the same
   trade. Note the eval cannot detect this by itself — golden and agent both escalate, so they
   agree; it shows up only when a reviewer reads the escalations and judges them unnecessary.
   If that is the verdict, the fix is to carve "plan how-to with no account lookup" out of the
   rule, not to re-label the rows.
5. **Row 13 — "there's a song on the new album that is listed but won't play… everyone else
   is unable to listen"** → `content_availability`, not `playback_issue`. The deciding fact is
   *whose* playback fails: everyone's, so it is a licensing/greyed-out catalogue row, and the
   brand answered with the licensing page. Row 119 (distorted audio on one machine) went the
   other way for the same reason.
6. **The iPhone X cluster — rows 12, 39, 92, 95, 116 and dev 29** → `feature_request`, not
   `app_bug`. The app runs; it has simply not been adapted to a new screen size. Six rows ride
   on this one ruling, and it is the single largest reason `feature_request` is 16.3% of the
   set. The brand's reply ("we're working on it, stay tuned") is a roadmap answer, not a
   diagnostic, which is the evidence for it.
7. **Rows 69 and 81 — "is this password-reset email real or phishing?" and "a text asked me
   to confirm a £9.99 subscription I never signed up for"** → escalate `safety_legal`, not
   `needs_account_access`. The intents differ (`account_access`, `billing_and_refund`) but
   both are fraud attempts aimed at the customer, and a confident wrong answer walks someone
   into a phishing link. These are the highest-harm rows in the set.
8. **Row 50 — "please review your choice of ads… lazy casual sexism"** → `other`, escalate
   `safety_legal`. A discrimination complaint about ad content has no support resolution, so
   the intent is `other`; the enum has no content-moderation category, and `safety_legal` is
   the honest nearest neighbour for a trust-and-safety report the brand DM'd to collect
   material on. Arguable — `out_of_scope` would also be defensible.
9. **Row 99 vs row 145 — both "I was logged out and my playlists are gone".** Row 99 adds that
   the username became a random string and Premium vanished: that is the signature of being in
   a *different* account, so `account_access`. Row 145 has none of that evidence, so its intent
   stays `library_playlist_issue`. Both escalate, but for different labels — 99 because the
   intent always does, 145 because the brand's reply was a bare DM deflect (rule (b)9). Dev
   row 3 follows row 99. This pair is the clearest case in the set of the separation the guide
   insists on: **the message decides the intent, the reply decides the escalation.**
10. **Row 76 — "Will my USA-based account work while I'm traveling in Europe?"** →
    `how_to_question` but escalated `needs_account_access`, because the correct answer depends
    on whether the account is Premium or free and the brand replied with a bare DM deflect. It
    is the only escalated `how_to_question` (1 of 4), and it is exactly the weakness the
    taxonomy flags for that intent (39.0% of its brand replies still deflect).

Three more sets of rows that were close and are worth knowing about: **rows 41, 139 and 142** (a concert
presale code, a request for live sessions, a podcaster asking how to publish) are all
`other`/`out_of_scope` under the surface test in tie-break 9 — ticketing, editorial
programming and the creator surface are not the Spotify product, and whether the brand
happened to post a helpful link does not change that; they sit with the other creator-surface
rows 3, 77 and 131. **Row 86** ("please review this complaint") was labelled
`ambiguous` even though the brand's reply reveals a student-discount thread, because the agent
never sees the reply and the message alone says nothing; **dev row 20** says "fraud" about
their own bank's incident, so it is `billing_dispute`, not `safety_legal` — the deterministic
rule disagrees there too.

## 5. Audit

The labels are the evaluation's ground truth, so they were checked rather than trusted. Every
invariant below is a test in `tests/test_golden_integrity.py`, so this section is reproducible with
`make test` rather than a claim to take on faith.

**Structure.** 160 rows, unique on `golden_id`, `pair_id` and `customer_tweet_id`; no duplicate
`customer_text`; no empty cell in any column; 40 dev rows disjoint from golden on both id columns.

**No leakage into the retrieval index.** Zero overlap with `corpus.csv` on `pair_id` (0/160), on
`customer_tweet_id` (0/160) and on verbatim `customer_text` (0/160). Every golden row is a first
turn. The day-level time split holds end to end: the corpus ends 2017-10-28 and the earliest golden
row is 2017-10-29, so nothing the agent is scored on could have been retrieved.

**Label contract.** Every `intent` is in the taxonomy and every `escalation_reason` is in
`rules.ESCALATION_CATEGORIES`. `escalation_reason == "none"` matches `should_escalate == False` on
all 160 rows. All four never-auto-handle intents — `account_access`, `billing_and_refund`,
`subscription_plan`, `other` — escalate without a single exception, as guide rules (b5)-(b7)
require. 78 of 160 rows escalate (48.8%), matching the distribution table above.

**Reason priority.** Each row's recorded reason was re-checked against the guide's (b) ordering,
which resolves ties by taking the first rule that fires. **No labelling error was found.** Two rows
looked like violations and are not: golden 70 (*"Your web player fucking blows."*) and golden 144
(*"... deactivated Facebook account . WTF"*) were being flagged `abusive` by `rules.pre_check`, but
guide rule (b2) explicitly excludes swearing *about the product* — golden 70 is almost verbatim the
guide's own counter-example. The labels were right and the **regex** was wrong.

That audit is what turned up the three guide/implementation mismatches recorded in decision log 12:
`abusive` over-firing on product swearing, `human_requested` implementing only half of rule (b4)
(it matched "speak to a human" but not "please DM me", which is how five golden rows ask), and
`safety_legal` missing the phishing and discrimination language rule (b1) names. After the fix,
`pre_check` fires on 8 golden rows, all 8 are true escalations, and the rule's reason agrees with
the hand-recorded reason on 8 of 8 — against 2 rows and 0 of 2 before.

## 6. Limitations

- **Single labeller, no inter-annotator agreement.** One pass, one judgement, AI-assisted.
  There is no second annotator and therefore no kappa; the honest confidence interval on any
  per-intent accuracy computed from these 160 rows is wider than the table suggests. The
  mitigation is transparency, not statistics: every row carries a note, and the guide is
  versioned next to the data.
- **Stratification bias.** The sample is stratified on the *weak* label, which over-samples
  messages that contain the weak labeller's keywords and under-samples the messages it misses.
  Since 45.6% of the weak labels turned out to be wrong, the golden set's intent distribution
  is **not** an estimate of true traffic prevalence — `content_availability` at 20.0% and
  `feature_request` at 16.3% are artefacts of stratify-then-relabel, not of Twitter.
- **Thin classes.** `playback_issue` (4), `how_to_question` (4) and `library_playlist_issue`
  (8) are too small for stable per-intent metrics; treat their accuracies as directional.
  The **dev slice contains no `app_bug` row at all** (the integrity test only requires the
  breadth on golden). This is tolerated because the thresholds tuned on dev
  (`CONFIDENCE_THRESHOLD`, `RETRIEVAL_THRESHOLD`) are **global, not per-intent**, so the gap
  costs calibration evidence rather than correctness for that intent; a larger dev slice is a
  one-more-week item, not a blocker.
- **Escalation is policy, not observation.** 48.8% of golden rows escalate largely because the
  policy escalates three whole intents plus `other`. Where the brand answered such a message in
  public (rows 0, 36, 106, 129, 142), the golden set disagrees with the brand *on purpose*. A
  reader who rejects that policy should re-read those notes before reading the eval numbers.
- **The reference reply is evidence, not truth.** Several replies are plainly wrong (row 101
  routes an account-compromise report to Ticketmaster; row 80 answers a bio typo with a device
  question; row 147 asks for a date of birth and the last four card digits in a DM). The
  labels follow the message for the **intent**; the reply is decisive only for **escalation**,
  under rule (b)9, which flipped six otherwise auto-handleable rows — 7, 43, 76, 108, 136, 145.
  Applying that rule consistently is the one thing a first labelling pass got wrong and a
  review round fixed: four `library_playlist_issue` rows had been left auto-handleable on the
  labeller's own view of what the published answer should have been.
- **Non-English rows.** Four messages are not in English — golden 130 (Italian), 133 and 137
  (Indonesian), dev 22 (Filipino). All four were labelled on content, with `non-English` in
  the note, per the guide: language is a property of a message, not an intent. The brand
  deflected three of them to language-specific email support, which the labels ignore.
- **No multi-label.** Several rows carry two complaints (row 63 access + family invite, row 64
  country + card, dev row 32 downloads + loyalty offers). Each got the primary intent, with
  the second named in the note; an eval that scores only the top intent will call a reasonable
  second-choice prediction wrong.
