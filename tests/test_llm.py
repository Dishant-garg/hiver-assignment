import json

import pytest

from support_agent.llm import (GPT_OSS_REASONING_EFFORT, CacheMissError, LLMClient, _groq_transport,
                               cache_key)


def fake_transport(calls):
    def t(model, messages, temperature, json_mode, max_tokens):
        calls.append(model)
        return json.dumps({"echo": messages[-1]["content"]})
    return t


def test_cache_miss_then_hit(tmp_path):
    calls = []
    c = LLMClient(tmp_path / "c.jsonl", offline=False, transport=fake_transport(calls), rpm=10000)
    msgs = [{"role": "user", "content": "hi"}]
    a = c.chat("m", msgs)
    b = c.chat("m", msgs)
    assert a == b and calls == ["m"] and c.stats == {"hits": 1, "misses": 1}
    # a new client reads the persisted cache
    c2 = LLMClient(tmp_path / "c.jsonl", offline=True, transport=None)
    assert c2.chat("m", msgs) == a


def test_offline_miss_raises(tmp_path):
    c = LLMClient(tmp_path / "c.jsonl", offline=True, transport=None)
    with pytest.raises(CacheMissError):
        c.chat("m", [{"role": "user", "content": "new"}])


def test_key_depends_on_params():
    m = [{"role": "user", "content": "x"}]
    assert cache_key("a", m, 0.0, True, 10) != cache_key("a", m, 0.7, True, 10)
    assert cache_key("a", m, 0.0, True, 10) != cache_key("b", m, 0.0, True, 10)
    assert cache_key("a", m, 0, True, 10) == cache_key("a", m, 0.0, True, 10)


def test_corrupt_cache_line_is_skipped(tmp_path):
    cache_path = tmp_path / "c.jsonl"
    msgs = [{"role": "user", "content": "hi"}]
    good_key = cache_key("m", msgs, 0.0, True, 400)
    with cache_path.open("w") as f:
        f.write(json.dumps({"key": good_key, "model": "m", "tag": "", "response": "ok"}) + "\n")
        f.write('{"key": "deadbeef", "model": "m", "tag": "", "resp\n')  # truncated line

    c = LLMClient(cache_path, offline=True, transport=None)
    assert c.chat("m", msgs) == "ok"
    with pytest.raises(CacheMissError):
        c.chat("m", [{"role": "user", "content": "unknown"}])


def _stub_openai_module(monkeypatch, recorded):
    """Replace openai.OpenAI with a stub whose .chat.completions.create records its kwargs.

    `_groq_transport` imports openai lazily inside the function body, so patching the attribute on
    the real module is enough and no network client is ever constructed.
    """
    openai = pytest.importorskip("openai")

    class _Message:
        content = '{"ok": true}'

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            recorded.append(kwargs)
            return _Response()

    class _Chat:
        completions = _Completions()

    class _StubClient:
        def __init__(self, **kwargs):
            self.chat = _Chat()

    monkeypatch.setattr(openai, "OpenAI", _StubClient)


def test_reasoning_effort_is_pinned_for_gpt_oss_only(monkeypatch):
    """The transport pins reasoning_effort for gpt-oss models and leaves every other family alone.

    gpt-oss bills reasoning tokens against max_tokens, so without the pin a 400-token ceiling
    truncates roughly a quarter of agent turns into `400 json_validate_failed`. The Qwen judge does
    not take the parameter in this form, so it must not be sent there.
    """
    recorded = []
    _stub_openai_module(monkeypatch, recorded)
    transport = _groq_transport("test-key", "https://example.invalid/v1")
    msgs = [{"role": "user", "content": "hi"}]

    transport("openai/gpt-oss-120b", msgs, 0.0, True, 400)
    transport("qwen/qwen3.8-27b", msgs, 0.0, True, 200)

    gpt_oss, qwen = recorded
    assert gpt_oss["reasoning_effort"] == GPT_OSS_REASONING_EFFORT
    assert "reasoning_effort" not in qwen
    # the pin must not disturb anything else the transport sends
    assert gpt_oss["model"] == "openai/gpt-oss-120b" and gpt_oss["max_tokens"] == 400
    assert gpt_oss["response_format"] == {"type": "json_object"}


def test_effort_tag_does_not_change_cache_key(tmp_path):
    """The effort tripwire is written to `tag` only; `cache_key` and cache hits are unaffected."""
    calls = []
    c = LLMClient(tmp_path / "c.jsonl", offline=False, transport=fake_transport(calls), rpm=10000)
    msgs = [{"role": "user", "content": "hi"}]
    c.chat("openai/gpt-oss-120b", msgs, tag="agent")
    c.chat("qwen/qwen3.8-27b", msgs, tag="judge")

    lines = [json.loads(l) for l in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert [r["tag"] for r in lines] == [f"agent@{GPT_OSS_REASONING_EFFORT}", "judge"]
    # the key on disk is still the plain cache_key, so a cache written before the tripwire still hits
    assert lines[0]["key"] == cache_key("openai/gpt-oss-120b", msgs, 0.0, True, 400)
    c2 = LLMClient(tmp_path / "c.jsonl", offline=True, transport=None)
    assert c2.chat("openai/gpt-oss-120b", msgs) == lines[0]["response"]
    assert c2.stats == {"hits": 1, "misses": 0}
