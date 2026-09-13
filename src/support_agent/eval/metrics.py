"""Automated metrics. Escalation gets a cost-weighted error because the two mistakes are not
symmetric: auto-handling something that needed a human (missed escalation) can harm the customer,
while escalating unnecessarily only costs agent time. miss_cost=5 is a stated assumption, not a fact."""
from __future__ import annotations

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def intent_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                 for i, l in enumerate(labels)}
    present = [l for l in labels if per_class[l]["support"] > 0]
    macro_f1 = sum(per_class[l]["f1"] for l in present) / max(1, len(present))
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class, "confusion": cm, "labels": labels}


def confusion_markdown(m: dict) -> str:
    labels = m["labels"]
    head = "| true \\ pred | " + " | ".join(labels) + " |\n|" + "---|" * (len(labels) + 1)
    rows = [f"| {l} | " + " | ".join(str(v) for v in row) + " |" for l, row in zip(labels, m["confusion"])]
    return "\n".join([head, *rows])


def escalation_metrics(y_true: list[bool], y_pred: list[bool], miss_cost: float = 5.0) -> dict:
    n = max(1, len(y_true))
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))   # missed escalation
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))  # unnecessary escalation
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "missed_escalation_rate": fn / n, "unnecessary_escalation_rate": fp / n,
            "automation_rate": sum(not p for p in y_pred) / n,
            "weighted_error": (miss_cost * fn + fp) / n, "n": len(y_true)}
