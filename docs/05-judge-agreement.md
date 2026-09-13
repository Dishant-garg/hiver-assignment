# Judge-vs-human agreement on 60 replies

`docs/04-judge-rubric.md` documents what the LLM judge is asked to do and which biases to expect.
This document is the evidence for whether its `overall` score can be believed. Sixty replies were
scored by hand against the same rubric, blind to the judge, and the two sets of scores compared.

Reproduce with:

```bash
.venv/bin/python scripts/sample_for_human_scoring.py   # writes data/golden/human_scoring_sheet.csv
.venv/bin/python scripts/judge_agreement.py            # writes results/judge_agreement.json
```

## Who the scorer was

**The 60 labels in `data/golden/human_judge_labels.csv` were produced by an AI assistant (Claude,
in a Claude Code session) applying `docs/04-judge-rubric.md` row by row, for review by the candidate.**
They are not the labels of a second independent human, and the limitations section below says what
that costs. Every label carries a one-line justification in the `note` column, so a reviewer can
disagree with any individual score and see exactly what it was based on.

## Protocol

1. `scripts/sample_for_human_scoring.py` drew 60 rows from `results/predictions.csv` with
   `config.SEED = 42`: 30 agent, 20 nearest-neighbour, 10 canned. Each `golden_id` appears once,
   so no customer message is seen twice with different candidates -- that would turn an absolute
   rubric score into a comparative ranking. The sheet it writes
   (`data/golden/human_scoring_sheet.csv`) has `sheet_id, golden_id, system, customer_text,
   reference_reply, candidate` and **no judge score, no judge rationale, no sub-scores**.
2. The rows were printed for scoring with the `system` column masked, so the scorer saw only the
   customer message, the brand's reference reply and the unlabelled candidate, and wrote a 1-5
   `overall` plus a note for each.
3. **`results/judge_scores.csv` was opened for the first time only after all 60 labels were
   written to disk**, when `scripts/judge_agreement.py` was run. The one earlier read of that file
   was by the sampler, which loads `usecols=["golden_id", "system"]` -- key columns only, no score
   or rationale -- because the judge run covers canned on `golden_id` 0-39 only and a human label
   with no judge score opposite it is wasted work.
4. `scripts/judge_agreement.py` joins on `(golden_id, system)` (`validate="one_to_one"`, and it
   raises rather than silently dropping a row that fails to match), calls
   `support_agent.eval.agreement.agreement_report`, and writes `results/judge_agreement.json`.

One deviation from a fully blind protocol, stated plainly: the first draw put 8 canned rows outside
the judged 0-39 range, so the sampler was corrected and those 8 rows were redrawn and scored
**knowing they were canned**. The leak is small by construction -- `canned_reply` is a single
constant string ("Sorry to hear that! Can you send us a DM with more details so we can take a
closer look?"), the scorer had already scored that exact string blind on two other rows, and the
only judgement left is whether it fits the customer's message. The 50 agent and nn rows, which is
where the interesting signal is, were never affected: they were scored blind and carried over
unchanged.

## Results

| slice | n | kappa (quadratic) | kappa 95% CI | Spearman | Spearman 95% CI | exact | within 1 | human mean | judge mean | judge - human |
|---|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|
| overall | 60 | 0.561 | [0.350, 0.711] | 0.591 | [0.372, 0.760] | 0.38 | 0.78 | 2.88 | 2.85 | -0.03 |
| agent | 30 | 0.571 | [0.339, 0.745] | 0.675 | [0.411, 0.845] | 0.43 | 0.80 | 3.53 | 3.17 | -0.37 |
| nn | 20 | 0.485 | [0.137, 0.751] | 0.583 | [0.199, 0.824] | 0.30 | 0.75 | 2.20 | 2.65 | +0.45 |
| canned | 10 | 0.155 | [-0.182, 0.623] | 0.274 | [-0.325, 0.861] | 0.40 | 0.80 | 2.30 | 2.30 | +0.00 |

Intervals are seeded percentile bootstraps (`config.SEED`, 2,000 resamples of the scored pairs,
2.5th/97.5th percentile), computed in `scripts/judge_agreement.py` and stored as `kappa_ci95` /
`spearman_ci95` in `results/judge_agreement.json`. Resamples whose score vector comes out constant
have an undefined kappa; `agreement.weighted_kappa` returns 0.0 there, so they pull the lower bound
down rather than being quietly dropped -- the conservative direction.

**The headline interval is wide: kappa 0.561 [0.350, 0.711] on n=60.** It excludes zero, so the
agreement is real, but it spans from "fair" to "substantial" on Landis-Koch. Any claim that needs
kappa to be above some threshold is not supported by this study; claims that need it to be above
zero are.

**Per-system kappa is not to be read as a measurement.** The bootstrap makes the point better than
prose: [0.339, 0.745] for agent, [0.137, 0.751] for nn, [-0.182, 0.623] for canned. The canned slice
also has almost no score variance on either side (human scores are 2s and 3s), which deflates kappa
arithmetically -- 0.155 there says "this slice is too small and too flat to measure", not "the judge
is random on canned replies". The `overall` row is the number worth quoting.

**Self-consistency** (`results/judge_consistency.json`, Task 14): the same judge re-scoring the same
replies at temperature 0.0 and 0.7 agrees with *itself* at kappa 0.704, Spearman 0.694, exact 0.55,
within-one 0.90. Two qualifications on using that as "the judge's own ceiling". It was measured on
**40 agent replies only** (the first 40 golden rows), so it is a ceiling for the agent slice, not for
nn or canned. And the like-for-like comparison is therefore not the overall 0.561 but the
**agent-slice kappa 0.571 [0.339, 0.745]** against self-consistency 0.704: the judge agrees with a
human on agent replies slightly less well than it agrees with itself, which is the expected ordering
-- if human agreement had come out *higher* than self-agreement, something would be wrong with the
study. The two intervals overlap heavily, so the gap between 0.571 and 0.704 is itself not
established.

**System ranking**, the thing the judge is actually used for:

| | 1st | 2nd | 3rd |
|---|---|---|---|
| human | agent 3.53 | canned 2.30 | nn 2.20 |
| judge | agent 3.17 | nn 2.65 | canned 2.30 |

Both put the agent clearly first, though not by the same distance: the human by 1.23 points over its
runner-up, the judge by 0.52 over its own. Neither can separate nn from canned: the two
baselines are 0.10 apart for the human and 0.35 apart for the judge, in opposite orders, on n = 20
and n = 10. All `parse_ok` values in the joined rows are true, so no clamped parse failure is
propping up any of these means.

## Five concrete disagreements

Judge rationales are quoted from `results/judge_scores.csv`; human notes from
`data/golden/human_judge_labels.csv`.

**1. H003 / golden 59 / nn -- human 1, judge 4 (judge too generous).**
Customer: "I'm already a premium subscriber. How do I say yes to the special 99$ yearly rate ending
12/31?" Candidate: *"Hi there! Just send it over, and we'd be happy to answer your question. /NQ"*
Human: "'Just send it over' refers to nothing; does not answer how to take the annual offer."
Judge: "grounded and friendly, but less specific than the reference because it does not explicitly
ask for the email address." **The human is right.** The judge noticed the missing ask and still
gave a 4; "send it over" has no antecedent at all, which makes the reply incoherent rather than
merely vague. The judge graded the style and missed that the sentence does not parse against the
question.

**2. H008 / golden 10 / nn -- human 2, judge 5 (judge too generous).**
Customer: "my account got charged twice in 5 mins... whyyyy?" Candidate: *"Hi! We've just sent you a
DM 🙂 /AR"* Human: "claims a DM was already sent, which may not be true, and collects none of the
information billing needs." Judge: "identical to the first piece of evidence provided, which is a
valid, on-brand, and actionable response." **The human is right, and the judge's reasoning shows
why it went wrong**: the reply scored 5 *because it matched the retrieved evidence verbatim*. A
retrieval baseline that copies a retrieved human reply is, by construction, always "grounded"; the
judge reads that as quality.

**3. H025 / golden 23 / agent -- human 3, judge 1 (judge too harsh).**
Customer: downloaded albums keep disappearing, Samsung S7. Candidate offers concrete steps: free up
space, allow background running, re-download. Human: "plausible download troubleshooting but the
settings path looks invented, and it misses the documented steps." Judge: "invents specific
troubleshooting steps ... not present in the provided evidence or reference." **Both have a point,
and the judge over-punished.** The steps are standard Android advice and mostly right; one menu path
is wrong. A 1 means "invents policy, steps, or promises" at the level of unusable; this reply would
half-help a customer, which is a 3.

**4. H044 / golden 38 / agent -- human 4, judge 2 (judge too harsh).**
Customer suggests a Snapchat filter showing the current track. Candidate: *"Great idea! We'll pass
your suggestion along to the team. Thanks for the creative input! 🎧"* Human: "friendly
acknowledgement of a feature idea; less actionable than the reference's vote link." Judge: "invents
a promise to 'pass the suggestion along' which is not supported by the evidence or reference, and
fails to provide the specific voting link." **Closer to a tie, leaning human.** "We'll pass it on" is
a routine brand courtesy the reference set uses constantly (H021, H055 and H056's references all
say a version of it), so calling it an invented promise is over-literal -- but the judge is right
that the missing vote link costs real actionability. A 3 would have satisfied both.

**5. H053 / golden 37 / canned -- human 2, judge 4 (judge too generous).**
Customer: "how long does it take you to get a new song on?? ... getting bored of this." Candidate is
the constant canned reply asking them to DM. Human: "catalogue/licensing question deflected to DM;
the reference's licensing explanation is what was needed." Judge: "grounded, friendly, and
actionable, though it misses the specific licensing context." **The human is right.** A DM cannot
change when a song is licensed, so the reply sends the customer down a dead end; "actionable" in
the rubric means a next step that helps, not any next step at all. This is the single failure mode
that most inflates the canned baseline.

The pattern across all 13 disagreements of 2 points or more is consistent and one-directional:
**17 judge rationales use the word "invent", and those rows average -1.06 (judge below human);
4 rationales praise the candidate for being identical to the evidence, and those average +1.50
(judge above human).** Every one of the 3 rows where the judge scored <= 2 and the human >= 4 is an
agent row; 4 of the 6 rows where the judge scored >= 4 and the human <= 2 are nn rows. The judge is
measuring fidelity to the retrieved evidence more than it is measuring whether a customer would be
helped -- exactly the "reference anchoring" bias `docs/04-judge-rubric.md` predicted, now with a
number on it. That bias is *against* the system under test and *for* the retrieval baseline, so it
does not flatter the agent's headline result.

## Verdict

**Trustworthy for ranking systems: yes, with one boundary.** kappa 0.561 [0.350, 0.711] and Spearman
0.591 [0.372, 0.760] on n=60 are **moderate** agreement on Landis-Koch (substantial starts at 0.61,
which is inside the interval but above the point estimate), the judge's mean (2.85) is within 0.03
of the human's (2.88) so there is no global leniency or severity offset, and both scorers put the
agent on top (the human by 1.23 points over its runner-up, the judge by 0.52 over its own). That is
enough to support the claim the evaluation actually makes
-- the LLM agent beats the retrieval and canned baselines on reply quality. **It is not enough to
rank the two baselines against each other**, for two independent reasons. First, human and judge
order nn and canned differently, the gaps are 0.10 and 0.35 points, and n is 20 and 10. Second, the
canned slice is drawn from golden ids 0-39, which are not representative of the golden set (see
Limitations), so its 2.30 describes canned-on-those-40-rows, not canned overall. Any claim of the
form "nn beats canned" is unsupported by this study.

**Trustworthy for scoring an individual reply: no.** Exact agreement is 0.38, so on nearly two rows
in three the judge's number is not the number a careful rubric-reader would give, and 13 of 60 rows
differ by 2 or more points -- the distance between "would half-help" and "unusable". The judge's own
score distribution is polarised (14 ones and 11 fives out of 60) against a human distribution that
concentrates in the middle (22 twos, 15 threes, 15 fours). Within-one agreement of 0.78 means the
judge is usually in the right neighbourhood, which is why the aggregate works while the individual
score does not. Do not cite a single judge score as evidence about a single reply, and do not use
the judge as an automatic gate on a production reply.

## Limitations

- **One scorer, and that scorer is an AI.** There is no second human, so there is no human-human
  agreement baseline to compare 0.561 against. The honest reading is that kappa 0.561 is an upper
  bound on nothing and a lower bound on nothing -- it is one number from one rater.
- **The scorer and the judge may share biases.** Both are large language models. Correlated error
  inflates agreement: where both are wrong in the same direction, this study records agreement and
  calls it evidence. A human rater who has worked a support queue would likely disagree with both,
  most plausibly on the "ask them to DM" replies that both scorers treat as acceptable.
- **n = 60**, and 30/20/10 within it. The overall figure is usable (kappa 0.561 [0.350, 0.711]);
  every per-system figure is indicative at best, and the canned kappa [-0.182, 0.623] is not
  interpretable at all.
- **The canned slice is not a random sample of the golden set.** It is drawn from golden ids 0-39,
  the only rows the judge scored for canned, and those 40 rows differ from the other 120: content
  availability is 35% of them against 15% of ids 40+, `playback_issue` and `how_to_question` are
  absent entirely, and the should-escalate rate is 0.40 against 0.517. Since `canned_reply` is a
  single constant string, its score is driven entirely by which customer messages it is answering
  -- and content-availability questions are exactly the messages a "please DM us" reply serves
  worst. The canned mean of 2.30 therefore describes **canned-on-golden-0-39**, and most likely
  understates canned on the full golden set. This is inherited from the judge run's budget decision
  (canned judged on 40 rows only), not from this study's sampling; the agent and nn slices are drawn
  from all 160 rows and are unaffected.
- **Only `overall` was compared.** `groundedness`, `correctness`, `tone` and `actionability` were
  never human-labelled, so nothing here licenses citing a sub-score (see rubric section 3).
- **Single pass.** The scorer labelled each row once, with no re-scoring after a gap, so there is no
  measure of the *human's* self-consistency to set against the judge's 0.704 -- which itself was
  measured on 40 **agent** replies only, and so compares like-for-like against the agent-slice kappa
  of 0.571, not against the overall 0.561.
