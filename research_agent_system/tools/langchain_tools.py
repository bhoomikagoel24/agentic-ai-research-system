"""
research_agent_system/tools/langchain_tools.py

Wraps your EXISTING retrieval functions as @tool-decorated callables,
so an LLM can decide which source to query and when it has "enough"
papers — instead of the research_agent.py v1 behavior of always
hitting both arXiv and Semantic Scholar for every query.

Nothing about *how* papers are fetched changes: fetch_from_arxiv and
fetch_from_semantic_scholar are imported unmodified from v1's
tools/arxiv_tool.py and tools/semantic_scholar_tool.py (same caching,
same retry/backoff). Only the decision of *when* to call them moves
from hardcoded Python into the LLM's tool-calling loop.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from research_agent_system.tools.arxiv_tool import fetch_from_arxiv
from research_agent_system.tools.semantic_scholar_tool import (
    fetch_from_semantic_scholar,
)
from research_agent_system.memory.memory_manager import MemoryManager
from research_agent_system.utils.logger import get_logger

logger = get_logger(__name__)

# One shared MemoryManager instance for all memory-related tool calls.
# (MemoryManager itself just opens a ChromaDB PersistentClient — cheap,
# but no reason to re-open it on every tool invocation.)
_memory = MemoryManager()


@tool
def search_arxiv(query: str, limit: int = 2) -> str:
    """
    Search arXiv for research papers matching a query.
    Returns a JSON-encoded list of papers, each with
    title, abstract, year, url, authors, source.
    Use this for preprints and CS/ML-heavy topics.
    """
    try:
        papers = fetch_from_arxiv(query, limit)
    except Exception as e:
        logger.warning(f"search_arxiv tool failed: {e}")
        papers = []
    return json.dumps(papers, ensure_ascii=False)


@tool
def search_semantic_scholar(query: str, limit: int = 2) -> str:
    """
    Search Semantic Scholar for research papers matching a query.
    Returns a JSON-encoded list of papers, each with
    title, abstract, year, url, authors, source.
    Use this for peer-reviewed / cross-discipline coverage.
    """
    try:
        papers = fetch_from_semantic_scholar(query, limit)
    except Exception as e:
        logger.warning(f"search_semantic_scholar tool failed: {e}")
        papers = []
    return json.dumps(papers, ensure_ascii=False)


@tool
def search_memory(query: str, top_k: int = 3) -> str:
    """
    Search persistent semantic memory (ChromaDB) for previously
    retrieved papers, summaries, or plans related to a query.
    ALWAYS call this first, before search_arxiv or
    search_semantic_scholar, to avoid burning API quota on
    something already researched in a prior run.
    Returns a JSON-encoded list of {document, similarity} hits.
    """
    try:
        hits = _memory.search_memory(query, top_k=top_k)
    except Exception as e:
        logger.warning(f"search_memory tool failed: {e}")
        hits = []
    return json.dumps(hits, ensure_ascii=False)


RESEARCH_TOOLS = [search_memory, search_arxiv, search_semantic_scholar]
