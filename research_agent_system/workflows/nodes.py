"""
research_agent_system/workflows/nodes.py

One function per agent, each taking the running ResearchState and
returning a partial dict that LangGraph merges in. This is the part
that actually answers "how does my architecture become a graph":

    v1                              v2
    -----------------------------    -----------------------------
    PlannerAgent(cfg).run(topic)  -> planner_node(state)
    ResearchAgent(cfg).run(plan)  -> research_node(state)
    SummarizerAgent(cfg).run(...) -> summarizer_node(state)
    SynthesisAgent(cfg).run(...)  -> synthesis_node(state)
    CriticAgent(cfg).run(...)     -> critic_node(state)
    FormatterAgent(cfg).run(...)  -> formatter_node(state)

Every PROMPT constant below is imported from the v1 agent files
UNCHANGED — the prompt engineering already done isn't being redone,
just re-hosted. The only genuinely new behavior is:
  - critic -> synthesis conditional loop (route_after_critic), which
    v1 only did once, inline, in research_pipeline.py.
  - research_node uses a tool-calling ReAct sub-agent instead of an
    unconditional "call both APIs" loop.
"""

from __future__ import annotations

import json
import time
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from research_agent_system.config.config import Config
from research_agent_system.schemas import ResearchPlan, PaperSummary, SynthesisOutput
from research_agent_system.schemas.graph_schemas import QueryEvaluation, CritiqueOutput
from research_agent_system.workflows.graph_state import ResearchState

from research_agent_system.llm.chat_models import get_chat_llm, get_structured_llm
from research_agent_system.tools.langchain_tools import RESEARCH_TOOLS

from research_agent_system.tools.search_utils import (
    filter_papers,
    rank_papers,
    remove_similar_titles,
)

from research_agent_system.memory.memory_manager import MemoryManager
from research_agent_system.utils.logger import get_logger

# Reuse the actual prompt text already written for v1 — only the
# calling convention (SystemMessage/HumanMessage + structured output)
# changes, not the prompt engineering itself.
from research_agent_system.agents.planner_agent import PLANNER_PROMPT, EVAL_PROMPT
from research_agent_system.agents.summarizer_agent import SUMMARY_PROMPT
from research_agent_system.agents.synthesis_agent import SYNTHESIS_PROMPT
from research_agent_system.agents.critic_agent import CRITIC_PROMPT
from research_agent_system.agents.formatter_agent import FORMATTER_PROMPT

logger = get_logger(__name__)
_memory = MemoryManager()

MAX_CRITIC_REFINEMENTS = 2  # caps the synthesis<->critic loop


# =========================================================
# PLANNER NODE
# =========================================================

def planner_node(state: ResearchState) -> dict:
    topic = state["topic"]

    # ---- memory reuse, same threshold v1 used in planner_agent.py ----
    for hit in _memory.search_memory(topic, top_k=1):
        if hit.get("similarity", 0) > 0.92:
            try:
                data = json.loads(hit["document"])
                if "search_queries" in data and "sub_questions" in data:
                    logger.info("Planner: reusing memory")
                    return {"plan": ResearchPlan(**data)}
            except Exception:
                pass

    plan_llm = get_structured_llm(ResearchPlan)
    plan = plan_llm.invoke([
        SystemMessage(content="You are a research planning agent."),
        HumanMessage(content=PLANNER_PROMPT.format(topic=topic)),
    ])

    eval_llm = get_structured_llm(QueryEvaluation)
    evaluation = eval_llm.invoke([
        SystemMessage(content="You are a research query evaluator."),
        HumanMessage(content=EVAL_PROMPT.format(
            queries=json.dumps(plan.search_queries, indent=2)
        )),
    ])

    queries = plan.search_queries if evaluation.is_valid else (
        evaluation.improved_queries or plan.search_queries
    )

    final_plan = ResearchPlan(sub_questions=plan.sub_questions, search_queries=queries)
    _memory.save_plan(topic, final_plan.model_dump())

    logger.info(f"Plan ready — {len(final_plan.search_queries)} queries")
    return {"plan": final_plan}


# =========================================================
# RESEARCH NODE  (tool-calling ReAct sub-agent)
# =========================================================

def make_research_node(cfg: Config, ablation=None):
    # Build the tool list based on which sources are enabled.
    # This is what makes single_source actually different — without
    # this, all configurations would call both arXiv and Semantic Scholar.
    from research_agent_system.tools.langchain_tools import (
        search_memory, search_arxiv, search_semantic_scholar
    )
    allowed_sources = ablation.sources if ablation else ["arxiv", "semantic_scholar"]
    disable_memory  = ablation.disable_memory if ablation else False

    tool_list = []
    if not disable_memory:
        tool_list.append(search_memory)
    if "arxiv" in allowed_sources:
        tool_list.append(search_arxiv)
    if "semantic_scholar" in allowed_sources:
        tool_list.append(search_semantic_scholar)

    source_instruction = (
        "Use search_arxiv only (Semantic Scholar is disabled for this run)."
        if "semantic_scholar" not in allowed_sources
        else "Check memory first via search_memory, then use search_arxiv "
             "and/or search_semantic_scholar."
    )
    memory_instruction = (
        "" if disable_memory
        else "Check memory first via search_memory to avoid wasting API calls "
             "on something already researched. "
    )

    research_agent = create_react_agent(get_chat_llm(), tool_list)

    def research_node(state: ResearchState) -> dict:
        plan: ResearchPlan = state["plan"]
        all_papers: list[dict] = []

        for q in plan.search_queries:
            result = research_agent.invoke({
                "messages": [
                    SystemMessage(content=(
                        f"You are a research retrieval agent. "
                        f"{memory_instruction}"
                        f"{source_instruction} Stop "
                        "once you have gathered a handful of relevant papers "
                        "for the query — do not call tools more than 4 times."
                    )),
                    HumanMessage(content=f"Find research papers for: {q}"),
                ]
            })

            for msg in result["messages"]:
                if isinstance(msg, ToolMessage) and msg.name in (
                    "search_arxiv", "search_semantic_scholar"
                ):
                    try:
                        all_papers.extend(json.loads(msg.content))
                    except Exception as e:
                        logger.warning(f"Could not parse tool result: {e}")

            time.sleep(random.uniform(1, 2))  # light pacing between queries

        # ---- same dedup/filter/rank pipeline v1 used, reused as-is ----
        unique: dict[str, dict] = {}
        for p in all_papers:
            key = p.get("url") or p.get("title", "").lower().strip()
            if key and key not in unique:
                unique[key] = p

        filtered = filter_papers(
            list(unique.values()), plan.search_queries,
            keep=max(60, cfg.topk * 4)
        )
        filtered = remove_similar_titles(filtered)
        ranked = rank_papers(filtered)
        papers = ranked[: cfg.topk]

        for p in papers:
            try:
                _memory.save_paper(p)
            except Exception as e:
                logger.warning(f"Paper memory save failed: {e}")

        logger.info(f"Research: {len(papers)} papers kept after filtering")
        return {"papers": papers}

    return research_node


# =========================================================
# SUMMARIZER NODE
# =========================================================

def make_summarizer_node(cfg: Config, ablation=None):
    summary_llm = get_structured_llm(PaperSummary)
    disable_memory = ablation.disable_memory if ablation else False

    def summarizer_node(state: ResearchState) -> dict:
        papers = state["papers"]
        to_process = papers[:2] if cfg.dev_mode else papers

        # ---- pass 1: memory reuse lookups are local vector-store calls,
        # not network calls — cheap, so sequential is fine here.
        # When disable_memory=True (no_memory ablation), skip the lookup
        # entirely — all papers must be re-summarised from scratch.
        # This makes the no_memory configuration genuinely different.
        reused_summaries: list[PaperSummary] = []
        needs_llm: list[dict] = []
        for p in to_process:
            reused = False
            if not disable_memory:
                for hit in _memory.search_memory(p["title"], top_k=1):
                    if hit.get("similarity", 0) > 0.85:
                        try:
                            data = json.loads(hit["document"])
                            if "source_title" in data:
                                reused_summaries.append(PaperSummary(**data))
                                reused = True
                                break
                        except Exception:
                            pass
            if not reused:
                needs_llm.append(p)

        # ---- pass 2: each paper's LLM call is independent of the
        # others, so run them concurrently instead of one-by-one. This
        # was the single biggest lever on summarizer latency — ~5
        # sequential ~35s calls (~3 min total) become roughly the time
        # of the single slowest call.
        def _summarize_one(p: dict) -> PaperSummary | None:
            try:
                abstract = " ".join(p["abstract"].split()[:600])
                summary = summary_llm.invoke([
                    SystemMessage(content="You are a research paper analysis agent."),
                    HumanMessage(content=SUMMARY_PROMPT.format(
                        title=p["title"], abstract=abstract
                    )),
                ])
                summary.source_title = p["title"]
                summary.source_url = p.get("url", "")
                summary.source_year = p.get("year")
                summary.confidence = (
                    "high" if summary.quality_score >= 8 else
                    "medium" if summary.quality_score >= 5 else "low"
                )
                try:
                    _memory.save_summary(p, summary.model_dump())
                except Exception as e:
                    logger.warning(f"Memory save failed: {e}")
                return summary
            except Exception as e:
                # One paper failing shouldn't take the rest down with it
                # — this didn't exist before since the loop was
                # sequential and any exception would have just crashed
                # the whole node.
                logger.warning(f"Summarization failed for '{p.get('title', '?')}': {e}")
                return None

        new_summaries: list[PaperSummary] = []
        if needs_llm:
            # 4, not 8 — with Top-K now able to reach 40, summarizing
            # everything at once would fire many concurrent LLM calls in
            # the same few seconds, which is exactly what trips a
            # free-tier requests-per-minute limit. 4 keeps a real
            # speedup over fully sequential without manufacturing the
            # rate-limit problem this is meant to avoid.
            with ThreadPoolExecutor(max_workers=min(2, len(needs_llm))) as pool:
                for result in pool.map(_summarize_one, needs_llm):
                    if result is not None:
                        new_summaries.append(result)

        summaries = reused_summaries + new_summaries

        # de-dup by source_title, same as v1's run_pipeline cleanup
        unique = {s.source_title.lower(): s for s in summaries}
        logger.info(f"Summarizer: {len(unique)} summaries")
        return {"summaries": list(unique.values())}

    return summarizer_node


# =========================================================
# SYNTHESIS NODE
# =========================================================

def synthesis_node(state: ResearchState) -> dict:
    summaries = state["summaries"]

    compressed = [{
        "title": s.source_title[:80],
        "problem": s.problem[:120],
        "method": s.method[:120],
        "key_findings": s.key_findings[:150],
        "limitations": s.limitations[:100],
        "core_tradeoff": s.core_tradeoff[:80],
        "quality_score": s.quality_score,
        "confidence": s.confidence,
    } for s in summaries]

    prompt = SYNTHESIS_PROMPT.format(
        summaries=json.dumps(compressed, indent=2, ensure_ascii=False)
    )

    # If we're looping back from the critic, steer the regeneration.
    critique = state.get("critique") or {}
    weaknesses = critique.get("major_weaknesses", [])
    if weaknesses:
        prompt += (
            "\n\nA prior critique flagged these weaknesses — address them "
            f"directly this time:\n{json.dumps(weaknesses[:3], indent=2)}"
        )

    synthesis_llm = get_structured_llm(SynthesisOutput, max_tokens=3000)
    synthesis = synthesis_llm.invoke([
        SystemMessage(content="You are an advanced AI research synthesis agent."),
        HumanMessage(content=prompt),
    ])

    logger.info("Synthesis complete")
    return {"synthesis": synthesis}


# =========================================================
# CRITIC NODE  (+ the conditional refinement loop)
# =========================================================

def critic_node(state: ResearchState) -> dict:
    synthesis: SynthesisOutput = state["synthesis"]

    critic_llm = get_structured_llm(CritiqueOutput)
    critique = critic_llm.invoke([
        SystemMessage(content="You are an expert AI research critic. Be brutally honest."),
        HumanMessage(content=CRITIC_PROMPT.format(
            synthesis=json.dumps(synthesis.model_dump(), indent=2, ensure_ascii=False)
        )),
    ])

    retries = state.get("critique_retries", 0) + 1
    logger.info(f"Critic pass {retries}: {len(critique.major_weaknesses)} weaknesses")
    return {"critique": critique.model_dump(), "critique_retries": retries}


def route_after_critic(state: ResearchState) -> Literal["synthesis", "formatter"]:
    """
    This is the piece v1 didn't have: the README's own "Known
    Limitations" section calls out that the critic evaluates but
    doesn't trigger a re-synthesis. This conditional edge is that
    missing loop, capped so it can't spin forever.
    """
    critique = state.get("critique", {})
    retries = state.get("critique_retries", 0)

    if critique.get("major_weaknesses") and retries < MAX_CRITIC_REFINEMENTS:
        return "synthesis"
    return "formatter"


# =========================================================
# FORMATTER NODE
# Paste this into research_agent_system/workflows/nodes.py
# replacing the existing formatter_node function
# =========================================================

def formatter_node(state: ResearchState) -> dict:
    synthesis: SynthesisOutput = state["synthesis"]
    critique = state.get("critique", {})

    # ── Compress before sending to LLM ───────────────────────────────────
    # Sending full SynthesisOutput JSON causes HTTP 413 Payload Too Large
    # on free-tier providers after large runs (10+ papers). The formatter
    # only needs the cross-paper findings — not every intermediate field.
    # This reduces payload ~60-70% without changing any research logic.
    # Retrieval, ranking, summarization, synthesis, critique = unchanged.
    def compress_synthesis(s) -> dict:
        if s is None:
            return {}
        d = s.model_dump() if hasattr(s, "model_dump") else (
            s if isinstance(s, dict) else {})
        return {
            "common_methods": d.get("common_methods", [])[:5],
            "method_comparisons": [
                # drop evidence_reasoning — verbose, not needed by formatter
                {k: v for k, v in item.items() if k != "evidence_reasoning"}
                for item in d.get("method_comparisons", [])[:4]
            ],
            "agreements":         d.get("agreements", [])[:5],
            "contradictions":     d.get("contradictions", [])[:4],
            "common_limitations": d.get("common_limitations", [])[:5],
            "research_gaps":      d.get("research_gaps", [])[:5],
            "emerging_trends":    d.get("emerging_trends", [])[:4],
            "future_directions":  d.get("future_directions", [])[:5],
            "final_insight":      d.get("final_insight", ""),
            "confidence":         d.get("confidence", "medium"),
        }

    def compress_critique(c) -> dict:
        if c is None:
            return {}
        d = c.model_dump() if hasattr(c, "model_dump") else (
            c if isinstance(c, dict) else {})
        return {
            "overall_quality":  d.get("overall_quality", 7),
            "confidence":       d.get("confidence", "medium"),
            # only top 3 weaknesses — enough for the formatter to address
            "major_weaknesses": d.get("major_weaknesses", [])[:3],
        }

    synthesis_compressed = compress_synthesis(synthesis)
    critique_compressed  = compress_critique(critique)

    llm = get_chat_llm()
    report = llm.invoke([
        SystemMessage(
            content="You are a skilled technical writer who explains research "
                    "clearly in plain, grammatically clean professional English."
        ),
        HumanMessage(
            content=FORMATTER_PROMPT.format(
                synthesis=json.dumps(
                    synthesis_compressed, indent=2, ensure_ascii=False
                ),
                critic_feedback=json.dumps(
                    critique_compressed, indent=2, ensure_ascii=False
                ) or "No critique available",
            )
        ),
    ])

    logger.info("Formatter complete")
    return {"final_report": report.content}