# Citations and credits

Everything this project borrowed, with the place in the code that uses it.

## Dataset

**Customer Support on Twitter.** Stuart Axelbrooke (Kaggle user `thoughtvector`). Roughly 2.8
million tweets from customer-support conversations between consumers and brand accounts, captured
in 2017. Used here for one brand, SpotifyCares.
<https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter>

`src/support_agent/data/download.py` fetches it from the public Kaggle download endpoint, which
needs no credentials. Refer to the Kaggle dataset page for the licence and terms that apply to
reuse.

**Hugging Face mirror, used as a fallback.** When the Kaggle endpoint fails, the downloader falls
back to a mirror of the same `twcs.csv`.
<https://huggingface.co/datasets/SunidhiSriram/twcs>

The fallback exists so a reviewer with no Kaggle account or a blocked endpoint can still rebuild
the processed data from scratch. The mirror is a third-party upload, not an official
redistribution, so the Kaggle page remains the authority on provenance and terms.

## Model inference

**Groq.** All LLM inference runs on Groq's OpenAI-compatible chat completions endpoint,
`https://api.groq.com/openai/v1`, on the free tier. Documentation and the live model list:
<https://console.groq.com/docs> and <https://console.groq.com/docs/models>. The free tier's
per-minute and per-day limits shaped three design choices: one LLM call per message, a persistent
response cache, and resumable runs. See `src/support_agent/llm.py` and decision log entry 5.

**Agent model: `openai/gpt-oss-120b`.** An open-weight model served by Groq. Model card:
<https://huggingface.co/openai/gpt-oss-120b>. Set as `config.AGENT_MODEL`, overridable by the
`AGENT_MODEL` environment variable.

**Judge model: `qwen/qwen3.8-27b`.** Model card: <https://huggingface.co/Qwen/Qwen3.8-27B>. Set as
`config.JUDGE_MODEL`. Chosen from a different model family than the agent on purpose: a judge that
shares a lineage with the thing it grades is more likely to share its blind spots than to catch
them. Both ids were **verified against Groq's live model list on 2026-09-11** and used for the
shipped run; decision log entry 5 records the check and the ids it ruled out.

## Python libraries

**scikit-learn.** TF-IDF vectorisation and cosine similarity in `src/support_agent/retrieve.py`;
logistic regression in `src/support_agent/baselines/intent.py`; precision, recall, F1 and the
confusion matrix in `src/support_agent/eval/metrics.py`; `cohen_kappa_score` in
`src/support_agent/eval/agreement.py`; KMeans in `scripts/explore_intents.py`.
Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine
Learning Research* 12, 2825-2830. <https://scikit-learn.org>

**pandas.** Every CSV in this repository is read and written with it.
McKinney, W. (2010). "Data Structures for Statistical Computing in Python." *Proceedings of the
9th Python in Science Conference*, 56-61. <https://pandas.pydata.org>

**NumPy.** Ranking, correlation and argsort in `src/support_agent/retrieve.py` and
`src/support_agent/eval/agreement.py`.
Harris, C. R. et al. (2020). "Array programming with NumPy." *Nature* 585, 357-362.
<https://numpy.org>

**openai Python SDK.** Used as the HTTP client for Groq's OpenAI-compatible endpoint, not against
OpenAI's own API. It is imported lazily inside `_groq_transport` in `src/support_agent/llm.py`, so
the test suite never needs it. <https://github.com/openai/openai-python>

**python-dotenv.** Loads `.env` in `src/support_agent/config.py`.
<https://github.com/theskumar/python-dotenv>

**pytest.** The test suite. <https://pytest.org>

**tabulate.** Markdown table output in `scripts/survey_brands.py`.
<https://github.com/astanin/python-tabulate>

## Methods

**Weighted kappa.** `agreement.weighted_kappa` uses quadratic weights so a 4-versus-5 disagreement
on a 1-5 rubric counts far less than a 1-versus-5.
Cohen, J. (1968). "Weighted kappa: Nominal scale agreement with provision for scaled disagreement
or partial credit." *Psychological Bulletin* 70(4), 213-220.

**LLM-as-judge.** The blinded rubric judge in `src/support_agent/eval/judge.py`, the practice of
decomposing a holistic verdict into named criteria, and the bias list in `docs/04-judge-rubric.md`
(length preference, self-preference, position and reference effects) follow this paper and the
agreement-with-humans methodology it sets out.
Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D.,
Xing, E. P., Zhang, H., Gonzalez, J. E., Stoica, I. (2023). "Judging LLM-as-a-Judge with MT-Bench
and Chatbot Arena." *NeurIPS 2023 Datasets and Benchmarks Track*. arXiv:2306.05685.
<https://arxiv.org/abs/2306.05685>

**TF-IDF term weighting.** The retrieval scheme in `src/support_agent/retrieve.py`.
Spärck Jones, K. (1972). "A statistical interpretation of term specificity and its application in
retrieval." *Journal of Documentation* 28(1), 11-21.

## AI assistance

The code, the tests and the documents in this repository were written with the help of an AI
coding assistant, which the assignment permits. The design, the decisions recorded in
`docs/07-decision-log.md`, and the review of every generated artefact are the candidate's own, and
every non-obvious choice is documented with its reasoning and its cost so that it can be
questioned.

Two places where this matters more than usual are flagged in the documents themselves. The golden
set labels were produced in an AI-assisted pass against a guide written beforehand, by a single
labeller, with no inter-annotator agreement figure, and the provenance note at the top of
`docs/03-golden-set.md` says so. The same applies to the human scores in the judge agreement
study, and `docs/05-judge-agreement.md` states it there.
