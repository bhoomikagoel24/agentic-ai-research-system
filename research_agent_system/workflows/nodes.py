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

def make_research_node(cfg: Config):
    research_agent = create_react_agent(get_chat_llm(), RESEARCH_TOOLS)

    def research_node(state: ResearchState) -> dict:
        plan: ResearchPlan = state["plan"]
        all_papers: list[dict] = []

        for q in plan.search_queries:
            result = research_agent.invoke({
                "messages": [
                    SystemMessage(content=(
                        "You are a research retrieval agent. Check memory "
                        "first via search_memory to avoid wasting API calls "
                        "on something already researched. Then use "
                        "search_arxiv and/or search_semantic_scholar. Stop "
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

        filtered = filter_papers(list(unique.values()), plan.search_queries)
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

def make_summarizer_node(cfg: Config):
    summary_llm = get_structured_llm(PaperSummary)

    def summarizer_node(state: ResearchState) -> dict:
        papers = state["papers"]
        to_process = papers[:2] if cfg.dev_mode else papers
        summaries: list[PaperSummary] = []

        for p in to_process:
            # ---- memory reuse by title, same threshold v1 used ----
            reused = False
            for hit in _memory.search_memory(p["title"], top_k=1):
                if hit.get("similarity", 0) > 0.85:
                    try:
                        data = json.loads(hit["document"])
                        if "source_title" in data:
                            summaries.append(PaperSummary(**data))
                            reused = True
                            break
                    except Exception:
                        pass
            if reused:
                continue

            abstract = " ".join(p["abstract"].split()[:250])
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

            summaries.append(summary)
            try:
                _memory.save_summary(p, summary.model_dump())
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

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

    synthesis_llm = get_structured_llm(SynthesisOutput)
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
# =========================================================

def formatter_node(state: ResearchState) -> dict:
    synthesis: SynthesisOutput = state["synthesis"]
    critique = state.get("critique", {})

    llm = get_chat_llm()
    report = llm.invoke([
        SystemMessage(content="You are a skilled technical writer who explains research "
                              "clearly in plain, grammatically clean professional English."),
        HumanMessage(content=FORMATTER_PROMPT.format(
            synthesis=json.dumps(synthesis.model_dump(), indent=2, ensure_ascii=False),
            critic_feedback=json.dumps(critique, indent=2, ensure_ascii=False) or "No critique available",
        )),
    ])

    logger.info("Formatter complete")
    return {"final_report": report.content}
