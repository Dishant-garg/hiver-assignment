# Evaluation summary
n = 160 golden examples

95% CIs are seeded percentile bootstraps over the golden rows (10,000 resamples, seed 42). They describe sampling noise at this n only -- not labelling error, and not the fact that this is one brand.

## Intent
| system | accuracy | 95% CI | macro-F1 | 95% CI |
|---|---|---|---|---|
| majority | 0.125 | [0.075, 0.181] | 0.022 | [0.014, 0.031] |
| logreg | 0.463 | [0.388, 0.537] | 0.463 | [0.364, 0.542] |
| agent | 0.738 | [0.669, 0.806] | 0.679 | [0.592, 0.755] |

## Escalation (positive = escalate)
| system | precision | recall | missed-esc rate | unnecessary-esc rate | automation rate | weighted error | 95% CI |
|---|---|---|---|---|---|---|---|
| always | 0.487 | 1.000 | 0.000 | 0.512 | 0.000 | 0.512 | [0.438, 0.588] |
| rules_only | 1.000 | 0.103 | 0.438 | 0.000 | 0.950 | 2.188 | [1.812, 2.562] |
| agent | 0.636 | 0.872 | 0.062 | 0.244 | 0.331 | 0.556 | [0.381, 0.756] |

## Reply quality (LLM judge, 1-5)
| system | n | grounded | correct | tone | actionable | overall | overall 95% CI | share overall>=4 | parse fail |
|---|---|---|---|---|---|---|---|---|---|
| agent | 160 | 3.21 | 3.26 | 4.53 | 3.33 | 3.06 | [2.819, 3.306] | 0.44 | 0 |
| canned | 40 | 2.85 | 2.10 | 3.83 | 2.27 | 2.12 | [1.825, 2.450] | 0.15 | 0 |
| nn | 160 | 4.95 | 2.64 | 4.38 | 2.94 | 2.66 | [2.406, 2.925] | 0.38 | 0 |

## Head-to-head, paired on the same rows

Each comparison scores both systems on identical rows, so the interval is on the *gap*.

- **Intent, agent vs logreg** (exact McNemar): agent alone correct on 59 rows, logreg alone correct on 15, p = 2.55e-07.
- **Intent, agent vs majority** (exact McNemar): agent alone correct on 107 rows, majority alone correct on 9, p = 2.00e-22.
- **Escalation weighted error, agent minus always**: +0.0437 [-0.1562, +0.2625]. Agent came out cheaper in 34% of resamples (lower cost is better).
- **Escalation weighted error, agent minus rules_only**: -1.6313 [-2.0438, -1.2188]. Agent came out cheaper in 100% of resamples (lower cost is better).
- **Judged overall, agent minus nn** (n = 160): +0.400 [+0.106, +0.688]. Agent scored lower in 0% of resamples (higher score is better).
- **Judged overall, agent minus canned** (n = 40): +0.675 [+0.175, +1.150]. Agent scored lower in 0% of resamples (higher score is better).
