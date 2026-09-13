"""Central configuration. Everything data-dependent or tunable lives here so the
report can cite one place, and so a reviewer can override via environment."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
CACHE_FILE = DATA_DIR / "cache" / "llm_cache.jsonl"
RESULTS_DIR = ROOT / "results"

# Chosen in Task 2 (brand survey); the decision log records why.
BRAND = os.getenv("BRAND", "SpotifyCares")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")  # id verified against the live Groq /models list on
# 2026-09-11: a different model family from the agent (Qwen vs OpenAI), and ~200K tokens/day as shown on the Groq limits
# page on 2026-09-11 — that figure is a published tier quota that can change without notice, not a property of the model,
# so re-check it rather than trusting this comment. Neither `qwen/qwen3-32b` nor the `llama-3.3-70b-versatile` fallback
# is served any more; of the two listed Qwen 27B ids, only qwen3.8 returns valid JSON inside the judge's 200-token budget
# (qwen3.6 spends it all on reasoning and 400s). See decision log entries 5 and 8.

# Escalation thresholds, tuned on the dev slice in Task 14 (never on golden). See decision log
# entry 7: the confidence gate is inert on dev (all six grid values tie), so 0.0 is the honest
# setting rather than a tuned one; the retrieval gate is the only one that moves the metric.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.0"))
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.3"))
MAX_REPLY_CHARS = 280
TOP_K = 3

SEED = 42
GOLDEN_SIZE = 160
DEV_SIZE = 40
CORPUS_MAX_PAIRS = 8000  # keeps corpus.csv under 10 MB and TF-IDF fit under a few seconds


def brand_dir() -> Path:
    return PROCESSED_DIR / BRAND


def offline() -> bool:
    """True when LLM calls must be served from cache only (grader reproduction path)."""
    return os.getenv("LLM_OFFLINE", "0") == "1"
