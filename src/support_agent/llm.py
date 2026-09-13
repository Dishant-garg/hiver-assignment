"""The only place that talks to an LLM.

Three jobs: (1) a persistent JSONL cache keyed by (model, messages, params) so graders can
reproduce results offline; (2) a fixed minimum-interval spacer between calls for the Groq
free tier (not a token bucket -- it just enforces a floor on the gap since the last call);
(3) clean handling of daily caps so runs can resume tomorrow instead of crashing."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from support_agent import config

Transport = Callable[[str, list[dict], float, bool, int], str]


class CacheMissError(RuntimeError):
    """Raised in offline mode when a prompt is not in the cache."""


class DailyLimitError(RuntimeError):
    """Raised when Groq reports a per-day cap; the caller should stop and resume later."""


def cache_key(model: str, messages: list[dict], temperature: float, json_mode: bool, max_tokens: int) -> str:
    payload = json.dumps({"model": model, "messages": messages, "temperature": float(temperature),
                          "json_mode": json_mode, "max_tokens": max_tokens}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


# `max_tokens` on Groq is a reasoning+output budget, not an output budget: at the default reasoning
# effort `openai/gpt-oss-120b` spends 215-405 completion tokens on one agent turn, so the agent's
# 400-token ceiling truncates roughly a quarter of rows and the API rejects the fragment with
# `400 json_validate_failed`. Pinning the effort to "low" puts every observed turn at 113-178
# completion tokens, which both fits the ceiling and keeps a 200-call run inside the free tier's
# ~200K tokens/day. It is a fixed transport constant rather than an env override on purpose: it is
# deliberately absent from `cache_key`, so letting it vary per run would let a cached response
# answer a prompt issued under a different reasoning budget. Same rationale as `max_retries=0`
# and `timeout=60` below. Only the gpt-oss family takes this parameter in this form.
GPT_OSS_PREFIX = "openai/gpt-oss"
GPT_OSS_REASONING_EFFORT = "low"


def _groq_transport(api_key: str, base_url: str) -> Transport:
    from openai import OpenAI, RateLimitError  # imported lazily so tests never need it

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=60)

    def call(model, messages, temperature, json_mode, max_tokens):
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if model.startswith(GPT_OSS_PREFIX):
            kwargs["reasoning_effort"] = GPT_OSS_REASONING_EFFORT
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                text = str(e)
                if re.search(r"per day|TPD|RPD|daily", text, re.I):
                    raise DailyLimitError(text) from e
                resp_obj = getattr(e, "response", None)
                headers = getattr(resp_obj, "headers", None)
                retry_after = headers.get("retry-after") if headers is not None else None
                wait = None
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = None
                if wait is None:
                    m = re.search(r"try again in ([\d.]+)(ms|s)", text)
                    wait = (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)) if m else 2.0 * (2 ** attempt)
                print(f"  429: sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(min(wait + 0.5, 90))
        raise RuntimeError("rate-limited 6 times in a row")

    return call


class LLMClient:
    def __init__(self, cache_path: Path = config.CACHE_FILE, *, api_key: str | None = None,
                 base_url: str = config.GROQ_BASE_URL, offline: bool | None = None,
                 rpm: int = 25, transport: Transport | None = None):
        self.cache_path = Path(cache_path)
        self.offline = config.offline() if offline is None else offline
        self._cache: dict[str, str] = {}
        self._load()
        self._transport = transport
        if self._transport is None and not self.offline:
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY missing; set it in .env or run with LLM_OFFLINE=1")
            self._transport = _groq_transport(key, base_url)
        self._min_interval = 60.0 / rpm
        self._last_call = 0.0
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0}

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        skipped = 0
        with self.cache_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]
                except (ValueError, KeyError):
                    skipped += 1
        if skipped:
            print(f"warning: skipped {skipped} unreadable cache line(s) in {self.cache_path}", file=sys.stderr)

    def _append(self, key: str, model: str, tag: str, response: str) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": model, "tag": tag, "response": response}, ensure_ascii=False) + "\n")

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.0, json_mode: bool = True,
             max_tokens: int = 400, tag: str = "") -> str:
        key = cache_key(model, messages, temperature, json_mode, max_tokens)
        if key in self._cache:
            self.stats["hits"] += 1
            return self._cache[key]
        if self.offline or self._transport is None:
            raise CacheMissError(f"offline and not cached: tag={tag} model={model}")
        with self._lock:
            wait = self._min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
        response = self._transport(model, messages, temperature, json_mode, max_tokens)
        self.stats["misses"] += 1
        self._cache[key] = response
        # Tripwire, not a cache input: `reasoning_effort` is set by the transport and is absent from
        # `cache_key` on purpose (see GPT_OSS_REASONING_EFFORT). Recording it in the human-readable
        # `tag` means a future change to the effort shows up as a diff in the committed cache rather
        # than silently reusing lines generated under a different reasoning budget. `tag` is never
        # read back by `_load`, so this cannot affect hit rates on the existing 602 lines.
        self._append(key, model, f"{tag}@{GPT_OSS_REASONING_EFFORT}" if model.startswith(GPT_OSS_PREFIX) else tag,
                     response)
        return response
