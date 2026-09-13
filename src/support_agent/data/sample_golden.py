"""Sample the golden evaluation set and a disjoint dev slice from the time-held-out pool.

Stratified by the keyword weak label so rare intents are represented (pure random sampling
would give a golden set dominated by the top two intents). The weak label is only used for
sampling; the golden label is assigned by hand afterwards."""
from __future__ import annotations

import sys

import pandas as pd

from support_agent import config, intents


def stratified_sample(holdout: pd.DataFrame, n: int, min_per_stratum: int, seed: int, exclude_ids: set[int]) -> pd.DataFrame:
    pool = holdout[(holdout.turn_index == 0) & (~holdout.pair_id.isin(exclude_ids))].copy()
    if "weak_label" not in pool:
        pool["weak_label"] = pool.customer_text.map(intents.weak_label)
    parts = []
    for _, grp in pool.groupby("weak_label"):
        parts.append(grp.sample(min(min_per_stratum, len(grp)), random_state=seed))
    base = pd.concat(parts) if parts else pool.iloc[0:0]
    if len(base) > n:
        n_strata = pool.weak_label.nunique()
        raise ValueError(f"min_per_stratum={min_per_stratum} across {n_strata} strata needs {len(base)} rows but n={n}")
    rest = pool.drop(base.index)
    fill = rest.sample(min(max(0, n - len(base)), len(rest)), random_state=seed)
    out = pd.concat([base, fill]).sample(frac=1, random_state=seed).head(n).reset_index(drop=True)
    return out


def _export(df: pd.DataFrame, path) -> None:
    out = df.rename(columns={"brand_reply": "reference_reply"})[
        ["pair_id", "customer_tweet_id", "customer_text", "reference_reply", "weak_label", "created_at"]]
    out.insert(0, "golden_id", range(len(out)))
    out.to_csv(path, index=False)


def main() -> None:
    holdout = pd.read_csv(config.brand_dir() / "holdout.csv")
    holdout["weak_label"] = holdout.customer_text.map(intents.weak_label)
    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden = stratified_sample(holdout, config.GOLDEN_SIZE, min_per_stratum=8, seed=config.SEED, exclude_ids=set())
    dev = stratified_sample(holdout, config.DEV_SIZE, min_per_stratum=2, seed=config.SEED + 1, exclude_ids=set(golden.pair_id))
    _export(golden, config.GOLDEN_DIR / "golden_unlabelled.csv")
    _export(dev, config.GOLDEN_DIR / "dev_unlabelled.csv")
    print(golden.weak_label.value_counts().to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
