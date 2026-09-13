"""Reproduces the per-intent dm_deflect% / resolution_step% table in docs/06-intent-taxonomy.md.

Weak-labels the first-turn corpus rows, then applies the same deflection-any and
resolution-step regexes `scripts/survey_brands.py` used to pick the brand.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from survey_brands import DEFLECTION_PATTERN, RESOLUTION_PATTERN  # noqa: E402

from support_agent import config, intents  # noqa: E402


def main() -> None:
    first = pd.read_csv(config.brand_dir() / "corpus.csv").query("turn_index == 0").copy()
    first["weak_intent"] = first.customer_text.map(intents.weak_label)
    names = [i.name for i in intents.INTENTS] + ([intents.OTHER] if all(i.name != intents.OTHER for i in intents.INTENTS) else [])
    rows = []
    for name in names:
        reply = first.loc[first.weak_intent == name, "brand_reply"].fillna("")
        rows.append({"intent": f"`{name}`", "n": len(reply),
                     "dm_deflect%": round(100 * reply.str.contains(DEFLECTION_PATTERN, case=False, regex=True).mean(), 1),
                     "resolution_step%": round(100 * reply.str.contains(RESOLUTION_PATTERN, case=False, regex=True).mean(), 1)})
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
