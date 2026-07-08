"""
research_agent_system/llm/chat_models.py
"""

from __future__ import annotations

import os
import time
import threading
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from research_agent_system.config.settings import GOOGLE_API_KEY, GROQ_API_KEY

PRIMARY_MODEL  = "gemini-2.5-flash"
FALLBACK_MODEL = "llama-3.3-70b-versatile"

# ── Rate limiter ──────────────────────────────────────────────────────────────
# Gemini free tier: ~10-20 RPM.  Groq free tier: 30 RPM.
# InMemoryRateLimiter doesn't reliably intercept SDK-internal retries, so we
# use a simple thread-safe sleep limiter that runs BEFORE every call.
class _SleepRateLimiter:
    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._last = 0.0
        self._lock  = threading.Lock()

    def wait(self):
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last = time.monotonic()

_GEMINI_LIMITER = _SleepRateLimiter(rpm=8)   # conservative below free 10-20 RPM
_GROQ_LIMITER   = _SleepRateLimiter(rpm=25)  # conservative below free 30 RPM


# ── LLM call counter (for benchmark metrics) ─────────────────────────────────
from langchain_core.callbacks import BaseCallbackHandler

class LLMCallCounter(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.call_count:     int = 0
        self.fallback_count: int = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.call_count += 1
        if serialized.get("id", [""])[-1] == "ChatGroq":
            self.fallback_count += 1

    def reset(self):
        self.call_count = self.fallback_count = 0


# ── BENCHMARK_MODE: set BENCHMARK_MODE=true in .env to use Groq as primary ──
# Groq (30 RPM free) is far better for long benchmark runs than Gemini (10-20 RPM).
# For interactive Streamlit use, Gemini stays primary (better quality).
_BENCHMARK_MODE = os.getenv("BENCHMARK_MODE", "false").lower() == "true"


class _RateLimitedChatGoogle(ChatGoogleGenerativeAI):
    """Wraps ChatGoogleGenerativeAI with a sleep-based rate limiter."""
    _limiter: _SleepRateLimiter = None

    def invoke(self, *args, **kwargs):
        self._limiter.wait()
        return super().invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        self._limiter.wait()
        return super().stream(*args, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        base = super().with_structured_output(*args, **kwargs)
        limiter = self._limiter
        class _Wrapped:
            def invoke(self_, *a, **kw):
                limiter.wait()
                return base.invoke(*a, **kw)
            def with_fallbacks(self_, fallbacks, **kw):
                return base.with_fallbacks(fallbacks, **kw)
        return _Wrapped()


def _primary_model(temperature: float = 0.2, max_output_tokens: int = 2048):
    if _BENCHMARK_MODE:
        # Groq as primary in benchmark mode — 30 RPM, no daily cap issues
        m = ChatGroq(
            model=FALLBACK_MODEL,
            api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_output_tokens,
            max_retries=4,
        )
        _GROQ_LIMITER.wait()
        return m
    m = ChatGoogleGenerativeAI(
        model=PRIMARY_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_retries=3,  # lower retries — we sleep before calls so retries rarely needed
    )
    # Monkeypatch wait into invoke/stream
    _orig_invoke  = m.invoke
    _orig_stream  = m.stream
    _limiter      = _GEMINI_LIMITER
    m.invoke  = lambda *a, **kw: (_limiter.wait(), _orig_invoke(*a, **kw))[1]
    m.stream  = lambda *a, **kw: (_limiter.wait(), _orig_stream(*a, **kw))[1]
    return m


def _fallback_model(temperature: float = 0.2, max_tokens: int = 2048):
    if _BENCHMARK_MODE:
        # In benchmark mode, Gemini becomes the fallback
        return ChatGoogleGenerativeAI(
            model=PRIMARY_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=3,
        )
    return ChatGroq(
        model=FALLBACK_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=4,
    )


def get_chat_llm(temperature: float = 0.2, max_tokens: int = 4096):
    primary  = _primary_model(temperature, max_output_tokens=max_tokens)
    fallback = _fallback_model(temperature, max_tokens=max_tokens)
    return primary.with_fallbacks([fallback])


def get_structured_llm(schema: type[BaseModel], temperature: float = 0.2, max_tokens: int = 2048):
    primary  = _primary_model(temperature, max_output_tokens=max_tokens).with_structured_output(schema)
    fallback = _fallback_model(temperature, max_tokens=max_tokens).with_structured_output(schema)
    return primary.with_fallbacks([fallback])


