# LLM-as-judge — rubric, blinding, and known biases

`src/support_agent/eval/judge.py` (`ReplyJudge`) scores a candidate reply against a customer
message, the brand's own reference reply, and the evidence the candidate was allowed to draw on.
This document records the rubric verbatim, how blinding is enforced, which field is compared to
human labels and why, and the biases a reader should discount for when reading judge scores.

## 1. Rubric (verbatim, `judge.RUBRIC`)

```
Score the candidate reply from 1 (bad) to 5 (excellent) on each criterion:
- groundedness: uses only facts/steps that appear in the evidence or reference; 1 = invents policy, steps, or promises.
- correctness: would resolve or correctly advance the customer's issue, judged against the reference reply; 1 = wrong or irrelevant.
- tone: friendly, concise, on-brand for Twitter support, no blame; 1 = rude, robotic, or rambling.
- actionability: gives the customer a concrete next step; 1 = says nothing useful.
- overall: your holistic judgement of whether this reply could be sent as-is.
Return JSON: {"groundedness": int, "correctness": int, "tone": int, "actionability": int, "overall": int, "rationale": "one sentence"}
```

Every score is an integer 1-5. `ReplyJudge.score()` clamps whatever the model returns into that
range (`_clamp`) so a malformed or out-of-range value (a stray `9`, a `0`, a non-numeric field)
degrades to the nearest valid score instead of corrupting the aggregate statistics. Parse
failures are flagged by `parse_ok=False` and must be reported alongside aggregate scores, since a
clamped `1` is otherwise indistinguishable from a genuine worst-quality reply.

## 2. Blinding rules

The judge is never told which system — baseline, retrieval, or LLM agent — produced the
candidate reply, and never sees a system name, model name, or vendor at all.
`build_judge_messages()` passes the judge exactly four pieces of information:

1. the customer's message,
2. the reference reply the brand actually sent,
3. the evidence the candidate was allowed to use (each evidence string truncated to 280 chars),
4. the candidate reply itself, unlabelled.

The system prompt only names the brand being reviewed for (`"You are a strict quality reviewer
for {BRAND} customer support replies."`); it carries no mention of which pipeline generated the
candidate. `tests/test_judge.py::test_judge_prompt_is_blind_to_system_identity` enforces this
mechanically: it serialises the built prompt to lowercase text and asserts none of
`baseline`, `llm`, `gpt`, `nearest`, `canned`, `system a`, `system b` appear anywhere in it —
catching both an accidental label (e.g. "the LLM's reply") and a giveaway phrase (e.g. "system
A produced"). Without this, the judge could develop a preference for a system by its label
rather than by the quality of what it wrote — an artifact of the evaluation setup, not a
property of the reply.

## 3. Why `overall` is the compared score

The rubric asks for five numbers, but only `overall` is what gets compared against human
agreement labels (`agreement.agreement_report`). `groundedness`, `correctness`, `tone`, and
`actionability` exist so the judge reasons about each dimension explicitly before giving a
verdict — decomposition is known to make LLM scoring more consistent than a single freeform
1-5 ask — but they are diagnostic, not the metric of record. `overall` is the one question a
human rater is actually answering when they label a reply ("is this good enough to send"), so
it is the only score whose distribution should line up with human judgement, and the only one
`weighted_kappa`/`spearman` should be computed against. Treating the four sub-scores as
independently meaningful would invite cherry-picking whichever one makes a system look best.

## 4. Known judge biases

These are documented so a reader discounts judge scores appropriately rather than treating them
as ground truth:

- **Length preference.** LLM judges tend to score longer, more elaborated replies higher even
  when a shorter reply is equally correct and more appropriate for a 280-character Twitter
  support channel. A candidate that pads its reply with extra hedging or explanation can score
  better on `tone`/`actionability` for reasons unrelated to quality.
- **Self-preference.** A judge model tends to rate replies more favorably when they resemble its
  own writing style or were produced by the same model family. Since the agent and the judge may
  share a provider (Groq-hosted open models), this is a live risk here, not a theoretical one —
  it is the main reason blinding by *identity* is not sufficient on its own and scores should be
  read as directional, not absolute.
- **Reference anchoring.** Giving the judge the brand's actual reply as a reference is
  deliberate — it grounds `correctness` — but it also means a candidate that phrases the same
  resolution differently from the reference can be marked down for `correctness` or
  `groundedness` even when it is factually right, and a candidate that echoes the reference's
  wording can be marked up regardless of whether the customer's specific situation warranted it.

## 5. What this buys, and what it doesn't

Agreement statistics (`agreement.agreement_report`) quantify how well the judge's `overall`
tracks a human rater on the same replies — that number is the evidence for (or against) trusting
the judge as a cheap proxy for the golden/dev evaluation, not an assumption baked into this
design. Judge scores should be reported alongside that agreement figure, never in isolation.
