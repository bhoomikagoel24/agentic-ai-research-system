"""
research_agent_system/llm/chat_models.py

V2 replacement for tools/llm_tool.py's call_llm()/call_groq() pair.

What this buys you over the v1 version:
  - The Gemini -> Groq fallback is no longer hand-rolled retry code,
    it's a single `.with_fallbacks([...])` composition that LangChain
    understands natively (and that LangSmith will trace as a single
    runnable with visible fallback branches).
  - Structured output (Planner / Summarizer / Synthesis / Critic) no
    longer goes through extract_json() + manual pydantic construction.
    `.with_structured_output(Schema)` does that for you, on both the
    primary AND the fallback model.

Nothing here talks to ChromaDB or arXiv/Semantic Scholar — this file
is *only* the "how do I get an LLM" layer. Tools live in
tools/langchain_tools.py, prompts stay where they already are (the
v1 agent files), orchestration lives in workflows/.
"""

from __future__ import annotations

from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from research_agent_system.config.settings import (
    GOOGLE_API_KEY,
    GROQ_API_KEY,
)

# Same model choices as v1's tools/llm_tool.py — only the plumbing changed.
PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "llama-3.3-70b-versatile"


def _primary_model(temperature: float = 0.2, max_output_tokens: int = 2048) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=PRIMARY_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=max_output_tokens,  # was unbounded — real cost/runaway-output risk
        max_retries=6,  # this is langchain-google-genai's own default, kept explicit
    )


def _fallback_model(temperature: float = 0.2, max_tokens: int = 2048) -> ChatGroq:
    return ChatGroq(
        model=FALLBACK_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,  # was unbounded
        max_retries=4,  # bumped from the default of 2 — Groq is now also the
                         # landing spot for concurrent summarizer calls, so it
                         # needs more headroom against transient rate limits
    )


def get_chat_llm(temperature: float = 0.2, max_tokens: int = 4096):
    """
    Plain free-text chat model (used by the Formatter, and by the
    ReAct research sub-agent). Gemini first, Groq automatically if
    Gemini raises (rate limit, timeout, empty response, etc) — and
    each of those already retries internally before giving up.
    max_tokens bounds output length/cost; 4096 covers a full report.
    """
    primary = _primary_model(temperature, max_output_tokens=max_tokens)
    fallback = _fallback_model(temperature, max_tokens=max_tokens)
    return primary.with_fallbacks([fallback])


def get_structured_llm(schema: type[BaseModel], temperature: float = 0.2, max_tokens: int = 2048):
    """
    Chat model bound to a Pydantic schema via with_structured_output.
    Replaces extract_json(raw, model=Schema) from utils/validators.py.

    Both the primary and the fallback are bound to the same schema,
    so the fallback is a true drop-in if Gemini fails mid-call.
    """
    primary = _primary_model(temperature, max_output_tokens=max_tokens).with_structured_output(schema)
    fallback = _fallback_model(temperature, max_tokens=max_tokens).with_structured_output(schema)
    return primary.with_fallbacks([fallback])
