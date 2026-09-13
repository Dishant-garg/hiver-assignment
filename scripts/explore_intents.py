"""Exploration used to define the taxonomy: top bigrams, KMeans clusters over TF-IDF,
and 15 random messages per cluster for manual reading. Output is read by a human; not a pipeline step."""
import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from support_agent import config

k = int(sys.argv[1]) if len(sys.argv) > 1 else 12
corpus = pd.read_csv(config.brand_dir() / "corpus.csv")
first = corpus[corpus.turn_index == 0]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words="english", sublinear_tf=True)
X = vec.fit_transform(first.customer_text)
km = KMeans(n_clusters=k, random_state=config.SEED, n_init=5).fit(X)
terms = vec.get_feature_names_out()
for c in range(k):
    idx = (km.labels_ == c)
    top = terms[km.cluster_centers_[c].argsort()[::-1][:12]]
    print(f"\n=== cluster {c} (n={idx.sum()}): {', '.join(top)}")
    for t in first[idx].customer_text.sample(min(15, idx.sum()), random_state=config.SEED):
        print("  -", t[:160])
