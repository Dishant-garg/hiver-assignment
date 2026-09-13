"""Intent baselines. Trivial: majority class. Simple: TF-IDF + logistic regression trained on
keyword weak labels from the corpus (no hand labels are used for training, so the golden set stays clean)."""
from __future__ import annotations

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from support_agent import config, intents


class MajorityIntent:
    def fit(self, labels: list[str]) -> "MajorityIntent":
        self.label = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [self.label] * len(texts)


class TfidfLogRegIntent:
    def __init__(self, seed: int = config.SEED):
        self.pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                  LogisticRegression(max_iter=2000, C=4.0, random_state=seed))

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogRegIntent":
        self.pipe.fit(texts, labels)
        return self

    def fit_weak(self, corpus_texts: list[str]) -> "TfidfLogRegIntent":
        return self.fit(corpus_texts, [intents.weak_label(t) for t in corpus_texts])

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.pipe.predict(texts))
