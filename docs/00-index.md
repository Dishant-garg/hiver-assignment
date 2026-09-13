# Documentation index

Files are numbered in the order they are worth reading. Each line says what that document answers,
so you can stop wherever you have what you came for.

| # | Document | What it answers |
|---|---|---|
| — | [`../README.md`](../README.md) | What this is, how to run it, and where every deliverable lives. Start here. |
| 01 | [`01-report.md`](01-report.md) | **The write-up.** Problem framing, the system, results against two baselines per task, the top-five failure analysis, what is misleading about the headline number, and what one more week would buy. |
| 02 | [`02-architecture.md`](02-architecture.md) | The runtime path and the offline pipeline as diagrams, plus a paragraph per component: responsibility, inputs, outputs, and why it exists. |
| 03 | [`03-golden-set.md`](03-golden-set.md) | How the 160 golden and 40 dev rows were sampled and labelled, the label distribution, the ten hardest calls with the ruling on each, and the limitations. |
| 04 | [`04-judge-rubric.md`](04-judge-rubric.md) | The LLM judge's rubric verbatim, how blinding is enforced and tested, why `overall` is the compared score, and the biases to discount for. |
| 05 | [`05-judge-agreement.md`](05-judge-agreement.md) | Whether the judge can be believed: kappa, Spearman, exact and within-one agreement, self-consistency, concrete disagreements, and a verdict on what it can and cannot be trusted to do. |
| 06 | [`06-intent-taxonomy.md`](06-intent-taxonomy.md) | The nine intents plus `other`: definitions, boundary rules for the three close pairs, why each `auto_handle_allowed` flag is what it is, and the taxonomy's known weaknesses. |
| 07 | [`07-decision-log.md`](07-decision-log.md) | The non-obvious decisions in the order they were made, each with what was chosen, why, and what it cost — including the ones that were later revised. |
| 08 | [`08-code-walkthrough.md`](08-code-walkthrough.md) | Every file in a table, a reading order for a newcomer, and recipes for changing things: adding an intent or a rule, swapping the model, moving a threshold, adding a metric, re-labelling a row. |
| 09 | [`09-citations.md`](09-citations.md) | The dataset, models, libraries and papers this project relies on, and a plain statement about AI assistance. |

## Generated artifacts, not prose

| Path | What it holds |
|---|---|
| [`../results/summary.md`](../results/summary.md) | Every headline metric with its bootstrap 95% CI, plus the paired head-to-head comparisons. Rewritten by `make reproduce`; `make verify` asserts it has not moved. |
| [`../results/failures.md`](../results/failures.md) | Every missed escalation, unnecessary escalation and low-scoring reply, with the evidence the agent saw. The source for report section 6. |
| [`../results/intent_confusion.md`](../results/intent_confusion.md) | The agent's intent confusion matrix. |
| [`../data/golden/labeling-guide.md`](../data/golden/labeling-guide.md) | The rules the labels were written against, fixed before labelling started: per-intent borderline rulings, the escalation rules, the tie-breaks, and how to treat the brand's own reply. |
