"""Lexical retrieval of similar historical exchanges. TF-IDF was chosen over embeddings
because Groq offers no embedding endpoint and a laptop-only fit keeps reproduction under 15 min.
The score is cosine similarity in [0,1]; the agent treats a low top score as weak evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Evidence:
    pair_id: int
    customer_text: str
    brand_reply: str
    score: float


class Retriever:
    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus.reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform(self.corpus.customer_text.fillna(""))

    @classmethod
    def from_csv(cls, path: Path) -> "Retriever":
        return cls(pd.read_csv(path))

    def top_k(self, query: str, k: int = 3) -> list[Evidence]:
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        idx = np.argsort(-sims)[:k]
        return [Evidence(int(self.corpus.pair_id[i]), str(self.corpus.customer_text[i]),
                         str(self.corpus.brand_reply[i]), float(round(sims[i], 4))) for i in idx]
