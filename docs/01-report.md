# AI customer-support agent for SpotifyCares

Every number here comes from `results/` and reproduces offline with `make reproduce`. Every failure
case cites a `golden_id` you can look up in `results/failures.md` and `data/golden/golden_set.csv`.

## 1. Problem framing

SpotifyCares answers public tweets. A useful agent for that queue has to do three things and refuse
to do a fourth.

It has to say what the customer is asking about, draft something a human could send unedited, and
decide whether a human is needed at all, with a reason an agent can act on. What it must not do is
auto-send anything touching an account, money, or a policy promise. A wrongly escalated ticket costs
an agent a minute; a wrongly automated one can cost a customer their subscription, and
`eval/metrics.py` prices that asymmetry at 5 to 1.

That last constraint is what the taxonomy encodes. `account_access`, `billing_and_refund`,
`subscription_plan` and `other` carry `auto_handle_allowed=False` in `intents.py`, so the agent
never sends a reply on those intents however confident it is. "Good" here means recall on escalation
first, automation rate second, reply quality third.

Deliberately not built, one line of reasoning each:

- **Multi-turn dialogue.** The evaluation unit is one message, so dialogue state would be untested
  code -- and it changes what "escalate" means.
- **Actually sending replies.** No credentials, and a submission that can post to a real brand
  account is a liability.
- **Fine-tuning, and embeddings.** 8,000 pairs will not beat a 120B model prompted with retrieved
  examples, and Groq exposes no embedding endpoint. Section 6 argues the second was a false economy.
- **Banking77, and a UI.** Banking intents are not streaming intents, so adding them would have
  measured transfer learning; a CLI and a results directory prove the rest in an hour, not a day.

## 2. System

One incoming message runs through a sandwich. Deterministic rules check the raw text first
(`human_requested`, `safety_legal`, `pii`, `abusive`). A TF-IDF retriever pulls the three most
similar historical exchanges from `corpus.csv`. One LLM call (`openai/gpt-oss-120b`,
`reasoning_effort="low"`, temperature 0) returns JSON with an intent, a self-reported confidence, a
draft reply, an escalate flag and a reason. Rules then check the output (`low_confidence`,
`weak_evidence`, `intent_not_auto_handleable`, `forbidden_promise`, `too_long`). If any rule fires,
or the model asked for a human, the decision is escalate and the reason is the rule reasons joined
into a sentence. One call rather than three, because intent, draft and escalation share the same
context and the free tier does not stretch to 480 calls per evaluation.

The full decision flow is drawn in `docs/02-architecture.md`, diagram 1.

## 3. Golden set

200 messages were drawn from the time-held-out slice only. The corpus is the earlier 75% of calendar
days, the holdout the later 25%, split by day so no thread straddles the line. Sampling takes first
turns only, stratified on a keyword weak label with a floor per stratum, because a uniform draw
would have been mostly catalogue questions and `other`. Every row was labelled by hand against
`data/golden/labeling-guide.md`, fixed before labelling started, with a written justification on all
200 rows. The weak label was wrong on 45.6% of them, which is the argument for hand labelling. The
result is `golden_set.csv` (160 rows, the ground truth) and `dev_set.csv` (40 rows, disjoint by
`pair_id`, the only slice thresholds were tuned on). Distribution, the ten hardest calls and the
limitations are in `docs/03-golden-set.md`.

The set was then audited against the guide rather than trusted, and the audit ships as tests
(`tests/test_golden_integrity.py`) so a reviewer can re-run it. No duplicate message; zero overlap
with the retrieval corpus on `pair_id`, `customer_tweet_id` or verbatim text; every row a first
turn; the time split intact. All four never-auto-handle intents escalate without exception, and
`escalation_reason` agrees with `should_escalate` on all 160 rows. Checking each reason against the
guide's priority order found **no labelling error** — but three places where `rules.pre_check`
contradicted the guide it implements (decision log 12), which section 6 mode 5 then pays for.

## 4. Results against baselines

n = 160. Tables copied from `results/summary.md`. Every comparison is also run **paired** -- both
systems scored on the same rows, bootstrapping the gap rather than each system separately. The
systems fail on largely the same messages, so treating their scores as independent would inflate
the uncertainty of the difference and hide two results that are real.

### Intent

| system | accuracy | 95% CI | macro-F1 | 95% CI |
|---|---|---|---|---|
| majority | 0.125 | [0.075, 0.181] | 0.022 | [0.014, 0.031] |
| logreg | 0.463 | [0.388, 0.537] | 0.463 | [0.364, 0.542] |
| agent | 0.738 | [0.669, 0.806] | 0.679 | [0.592, 0.755] |

The agent is 27 points above TF-IDF logistic regression, and this is the one headline gap large
enough that sampling noise is not a candidate explanation: the intervals do not overlap, and on the
paired test the agent is alone correct on 59 rows against logreg's 15 (exact McNemar, p = 2.6e-07).
The gap is real but partly an artefact of how the baseline was trained: logreg learns from the
*weak* labels, which disagree with the hand labels on 45.6% of rows, so it reproduces a signal that
is itself 46% noise. Cross-validated on the golden labels it would score higher. Macro-F1 also
hides two unmeasurable classes: `playback_issue` and `how_to_question` have four rows each, so
their F1 moves 0.25 per row.

### Escalation (positive = escalate)

| system | precision | recall | missed-esc rate | unnecessary-esc rate | automation rate | weighted error | 95% CI |
|---|---|---|---|---|---|---|---|
| always | 0.487 | 1.000 | 0.000 | 0.512 | 0.000 | 0.512 | [0.438, 0.588] |
| rules_only | 1.000 | 0.103 | 0.438 | 0.000 | 0.950 | 2.188 | [1.812, 2.562] |
| agent | 0.636 | 0.872 | 0.062 | 0.244 | 0.331 | 0.556 | [0.381, 0.756] |

This is the table the project turns on, and **it is the one table that does not resolve.** The
agent's 0.556 is nominally worse than always-escalate's 0.512, but the paired gap is +0.0437
[-0.156, +0.263], with the agent cheaper in 34% of resamples -- an interval nearly ten times wider
than the difference inside it. An earlier draft read this as "the agent loses"; that was
over-reading a point estimate. At n=160 the two are **indistinguishable**.

What survives is the shape of the trade, not a winner: the agent misses 10 of 78 true escalations,
which always-escalate cannot do, and always-escalate automates nothing, which the agent's 33.1%
cannot be bought back from at any price. Choosing between them is a policy question about
`miss_cost` (section 7), not one this evaluation settles. Rules-only is the genuine loser and the only
decisive comparison here: 70 misses, gap -1.631 [-2.044, -1.219], cheaper in 100% of resamples. A
keyword layer alone is not a safety mechanism, even after correction against the guide (log 12).

### Reply quality (LLM judge, 1-5)

| system | n | grounded | correct | tone | actionable | overall | overall 95% CI | share overall>=4 | parse fail |
|---|---|---|---|---|---|---|---|---|---|
| agent | 160 | 3.21 | 3.26 | 4.53 | 3.33 | 3.06 | [2.819, 3.306] | 0.44 | 0 |
| canned | 40 | 2.85 | 2.10 | 3.83 | 2.27 | 2.12 | [1.825, 2.450] | 0.15 | 0 |
| nn | 160 | 4.95 | 2.64 | 4.38 | 2.94 | 2.66 | [2.406, 2.925] | 0.38 | 0 |

The agent leads, and the lead survives the paired bootstrap: +0.400 [+0.106, +0.688] against
nearest-neighbour, +0.675 [+0.175, +1.150] against canned on the 40 rows where both were judged,
ahead in 100% of resamples both times. The per-system intervals overlap, which is why the pairing
is necessary: the difference is measured far more precisely than either mean. Section 5 confirms
the ranking against a human scorer. Read the nn row
carefully. Its groundedness of 4.95 is degenerate: the baseline returns a retrieved historical reply
verbatim and the judge is shown that same retrieval as the evidence, so the candidate *is* the
evidence and cannot be ungrounded. Correctness 2.64 is the honest column there, and the agent beats
it by 0.63. The agent's own weakest dimensions are groundedness and correctness, both near 3.2,
against tone at 4.53. It sounds like the brand and is right about two-thirds of the time.

## 5. Is the judge worth believing

`docs/05-judge-agreement.md` is the evidence. Sixty replies were scored independently against the
same rubric, blind to the judge and with the system column masked; 58 are still comparable, two
having been redrafted by the retrieval fix after they were scored, and those are dropped rather
than compared across different text. Quadratic-weighted kappa is 0.539 [0.331, 0.697], Spearman
0.566 [0.336, 0.740], exact agreement 0.38, within-one 0.78. Judge mean 2.91 against human mean
2.93, so there is no global leniency offset. The same judge re-scoring at temperature 0.0 and 0.7
agrees with itself at kappa 0.704 on 40 agent replies -- the right comparison for the agent-slice
kappa of 0.539, not for the overall 0.539.

The verdict is narrow. Both scorers put the agent first, though not by the same distance: the human
by 1.29 points over its runner-up (canned at 2.30), the judge by 0.50 over its own (nn at 2.74).
Ranking is what the judge is used for, and the ranking holds. Neither scorer separates nn from
canned, so that comparison is unsupported, and no single judge score should be cited about a single
reply when exact agreement is 0.38.

The disagreements matter more than the kappa, because they are one-directional. Sixteen judge
rationales use the word "invent", and those rows average 1.06 points *below* the human. Four praise
a candidate for matching the evidence exactly, and those average 1.50 points *above*. The judge is
measuring fidelity to retrieved text rather than whether a customer is helped. That bias runs
against the agent and for the retrieval baseline, so the agent's 3.06 against nn's 2.66 is, if
anything, understated. One qualification: the human scorer was an AI assistant applying the written
rubric, not a second person, so scorer and judge can be wrong in the same direction, and this study
would record that as agreement.

## 6. Top five failure modes

Full cases in `results/failures.md`, generated by `python -m support_agent.eval.failures`.

**1. The agent invents product facts and UI paths when retrieval gives it nothing.** Golden 28,
"how do i cancel my spotify?", got back *"Open Settings > Account > Your Plan and tap 'Cancel
Premium.'"* Golden 73 invented a proration policy for the annual plan; golden 142 invented
`podcasters.spotify.com` and an RSS submission flow. Golden 7 asserted that Spotify no longer
supports Facebook sync. All four scored 1 or 2 from the judge, and all four are labelled
`should_escalate=True`. The evidence column shows why. For golden 7 the three retrieved replies are
*"What's happening exactly?"*, *"Can you try the steps here"*, and a question about 3G versus WiFi.
TF-IDF over 60-character tweets retrieves the brand's tone, not its content, so the prompt carries
no product facts and the model falls back on what it remembers about Spotify. Fix: retrieve
help-centre articles instead, and treat any step absent from the evidence as a reason to escalate.

**2. An intent error into an auto-handleable class is a silent safety failure.** Golden 28 and 73
are labelled `subscription_plan`, which is not auto-handleable; the agent called both
`how_to_question`, which is. Golden 139 and 142 are labelled `other` for `out_of_scope`; the agent
filed them as `feature_request` and `how_to_question`. Those four misses would have been caught by a
correct intent prediction. The gate reads `auto_handle_allowed` on the *predicted* intent, so intent
accuracy and escalation safety are one number wearing two hats. The sharper half of this mode is the
other five misses: golden 7, 76, 108, 136 and 145 have a true intent that is itself auto-handleable,
so a perfect classifier would have auto-handled them too. Fix: check the top two predicted intents,
and stop treating intent as a sufficient statistic for routing.

**3. The agent writes "DM us" and still marks the row handled.** Golden 145: *"was logged out of my
account and now all my playlists have gone?!!"* The reply is *"Please DM us your account email or
username so we can look into your missing playlists."* The judge gave it 5 on every dimension. The
agent set `escalate=False`. Asking a customer to move to DM *is* a handover, and the labelling guide
counts it as one (rule b9), so this reply is at once the best thing the agent wrote and a missed
escalation. Golden 28 does the same thing, closing its invented cancellation walkthrough with "Need
more help? DM us!", and golden 73 ends "DM us your email!". Nine auto-handled drafts request a DM: three (28, 73, 145)
are missed escalations, the other six are correct auto-handles. Fix: a post-check rule treating any
draft that asks for a DM or a channel switch as an escalation. It catches 3 of the 10 misses and
costs 6 correctly-automated rows, 11.3% of the automated pile, taking weighted error from 0.556 to
0.500 and automation from 0.331 to 0.275 -- a counterfactual on the evaluation set, and still well
inside the [-0.156, +0.263] band, so it moves the point estimate without settling anything.

**4. The retrieval gate does most of the escalating, mostly on the wrong rows.** 61 of the agent's
107 escalations involve `weak_evidence`, and 32 of the 39 unnecessary escalations are that rule
firing on its own. Golden 12, a plain iPhone X feature request the judge scored 4, escalated at a
top retrieval score of 0.217; golden 13 at 0.214. The median top score is 0.319, barely above the
0.3 threshold, so the rule thresholds vocabulary overlap rather than question difficulty.
Recomputing the escalation metrics with `weak_evidence` removed:
weighted error drops to 0.481 and automation rises to 0.556, misses going from 10 to 14. Post-hoc on
the evaluation set and inside the same noise band, so: a hypothesis, not a result. What makes it
worth acting on is the mechanism rather than the number -- the gate fires on vocabulary overlap,
which is worth re-tuning wherever it lands.

**5. `other` is a routing decision wearing an intent's clothes.** It is the catch-all and it is
non-auto-handleable, so it turns up on both sides of the ledger. Golden 1 (a left-handed layout
complaint), 116 (iPhone X update timing), 11 and 18 were all filed as `other` and escalated
needlessly; the judge scored golden 1 a 5. Meanwhile golden 139 and 142 are labelled `other` for
`out_of_scope`, the agent filed them elsewhere, and they auto-handled. Golden 70 is the sharpest case:
*"Your web player fucking blows."* is `other`/`ambiguous` by the guide — a complaint naming a
product surface with nothing to act on — and the agent calls it `app_bug` and drafts
troubleshooting for a problem nobody described. It is the tenth missed escalation, and it was
hidden until the rule audit: a faulty `abusive` regex had been escalating it for the wrong reason,
so a classification failure sat behind a rule failure. One label is being asked to
mean both "I do not know" and "a human must look at this". Fix: split it into `out_of_scope` and
`unclear`, both escalating, and let the model say which.

## 7. What is misleading about my headline number

The headline is intent accuracy 0.738 against 0.463. Here is what it hides.

**The headline is the one result that is not fragile, and it is not the interesting one.** Nobody
deploys a support agent because it classifies well; they deploy it because it is safe to let it
answer. That metric is escalation -- and on escalation this evaluation settles nothing. The gap to
always-escalate is +0.0437 [-0.156, +0.263], the agent cheaper in 34% of resamples, an interval
nearly ten times wider than the difference inside it. The system this architecture exists to justify
is, on its own primary safety metric, **not measurably different from a one-line baseline**: weaker
than winning, and a different claim from losing. Separating an effect this size would need
thousands of labelled rows, not 160.

**And that comparison rests on a constant I chose.** `miss_cost=5.0` is an a priori guess in
`eval/metrics.py`. The break-even is **4.30**. The agent wins for any `miss_cost` below it and loses
above it. At 4.0 the agent is ahead, 0.494 against 0.512; at 5.0 it is behind, 0.556 against 0.512.
A 14% change in a number nobody measured flips the ranking. So the ordering is doubly undetermined: the data cannot
separate the two systems, and the constant that would separate them is a guess.

**One of the two tuned knobs does nothing.** The confidence gate ships at 0.0. All six values in the
grid produced byte-identical metrics, because the agent reports confidence of at least 0.9 on 154 of
160 rows. The retrieval threshold of 0.3 won by one dev row out of 40, and 0.3 is the top of the
grid, so the search stopped at its own boundary. Read the configuration as "retrieval gate on,
confidence gate off", not as two calibrated numbers.

**The golden set is one AI-assisted labeller.** No second annotator, so no inter-annotator kappa.
The weak labeller disagreed on 45.6% of rows, which shows both that the hand labelling did work and
how much room there is to disagree.

**The intent distribution is not traffic.** Sampling stratifies on the weak label, so
`content_availability` at 20% and `feature_request` at 16% are artefacts of stratify-then-relabel.
Per-class numbers do not transfer to a production queue, and macro-F1 weights a four-row class like
a thirty-two-row one.

**The time split does not stop template leakage.** It stops thread leakage. SpotifyCares reuses the
same phrases for months, so the retriever can still find a near-identical reply to a message it has
never seen. Golden 145's top score is 0.738. That inflates the nn baseline and the agent's
groundedness alike.

**The judge is an LLM with a measured bias and moderate agreement.** kappa 0.539 [0.331, 0.697],
exact agreement 0.38. The reply gap of +0.400 [+0.106, +0.688] does survive *sampling* noise, but
sampling noise is not the binding constraint: a judge that matches a careful human exactly on 38%
of rows is being asked to resolve four tenths of a point.

**The intervals cover sampling noise and nothing else.** Every CI here answers one question: how
much would this number move on a different 160 rows from the same distribution. None covers
labelling error, the stratify-then-relabel distribution, the choice of `miss_cost`, one brand, or
one model at one reasoning effort. Those are the larger error terms and they have no interval at
all. A narrow CI is not a claim that a number is trustworthy -- only that re-sampling rows is not
what would break it.

**Every agent number was measured at `reasoning_effort="low"`.** The pin exists because a
default-effort run costs roughly 212K tokens against a ~200K daily cap, and Groq bills reasoning
tokens against `max_tokens`; about 25% of calls truncated before the pin went in. Groundedness at
3.21 is the agent's weakest judged dimension, and some of that may be reasoning the model was never
given room to do. The alternative has never been measured.

**"Automation rate" says nothing about the customer.** 33.1% means 33.1% of drafts went out without
a human, not that anyone was helped. Golden 7 auto-sent an invented policy claim, and it counts as
automation.

**The cache freezes one model version.** `make verify` replays the 520 committed responses this
evaluation used, 602 in the file overall, in about 9 seconds with zero cache misses, and asserts
the output is byte-identical. That proves the pipeline is deterministic. It proves nothing about
`gpt-oss-120b` next month, and determinism is not accuracy.

## 8. With one more week

1. **Re-tune or remove the retrieval gate** (mode 4). Section 6's diagnostic suggests dropping it
   lowers weighted error to 0.481 and raises automation from 0.331 to 0.556. Do it properly: extend
   the dev slice to 150 rows and the grid past 0.3, then measure on golden once.
2. **Add the DM-request post-check rule** (mode 3). It catches 3 of the 10 misses and costs 6
   correctly automated rows, taking weighted error from 0.556 to 0.500 and automation from 0.331 to
   0.275. It is the cheapest item here -- a deterministic rule, not a model change -- and the only
   one that moves the point estimate to the better side of always-escalate, though section 7 is the
   reason that is worth less than it sounds.
3. **Retrieve from help-centre articles instead of tweets** (mode 1). Past tweets carry tone and no
   facts. An article index plus a "cite or escalate" instruction attacks the agent's weakest
   dimension.
4. **Gate on the top two intents and split `other`** (modes 2 and 5). Both are taxonomy changes
   rather than model changes, and both are measurable against the golden set that already exists.
   Neither touches the five misses whose true intent is auto-handleable, which need an escalation
   signal that does not come from the intent at all.
5. **Get a second labeller onto 60 rows** (section 7). Everything above rests on 160 labels from one
   rater. An inter-annotator kappa would say which of these numbers are worth acting on and which
   sit inside the labelling noise.
6. **Label for power, not coverage** (section 7). The comparison that matters is the one n=160
   cannot resolve. Before tuning anything further, work out how many rows would actually separate
   the agent from always-escalate at this effect size -- and if that is unaffordable, change the
   metric to something cheaper to estimate, such as precision at a fixed recall floor.
