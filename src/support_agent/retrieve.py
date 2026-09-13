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

    # Ties are common and must break the same way everywhere. 16 of the 160 golden queries have
    # two or three neighbours at an identical cosine score, and np.argsort defaults to an unstable
    # quicksort, so which tied row came first depended on the BLAS underneath -- Accelerate on
    # macOS, OpenBLAS on Linux. That reordered the evidence, which reordered the prompt, which
    # changed the LLM cache key: the committed cache replayed on the machine that recorded it and
    # missed in CI. Rounding before the sort collapses last-bit differences, and kind="stable"
    # then breaks the remaining ties by corpus position, which is identical on every platform.
    _TIE_PRECISION = 6

    def top_k(self, query: str, k: int = 3) -> list[Evidence]:
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        idx = np.argsort(-np.round(sims, self._TIE_PRECISION), kind="stable")[:k]
        return [Evidence(int(self.corpus.pair_id[i]), str(self.corpus.customer_text[i]),
                         str(self.corpus.brand_reply[i]), float(round(sims[i], 4))) for i in idx]
