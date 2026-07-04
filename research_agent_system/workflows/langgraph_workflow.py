"""
research_agent_system/workflows/langgraph_workflow.py

This file existed in the repo already (as an empty placeholder under
"What's Next" -> LangGraph orchestration). This is that placeholder,
filled in.

    START -> planner -> research -> summarizer -> synthesis -> critic
                                                       ^             |
                                                       |             v
                                                       +---(loop)--- |
                                                                     v
                                                                 formatter -> END

The critic -> synthesis edge is conditional (route_after_critic):
loops back if the critique found weaknesses, up to
nodes.MAX_CRITIC_REFINEMENTS times, otherwise proceeds to formatter.

A MemorySaver checkpointer is attached so a run can be resumed by
thread_id if it crashes mid-pipeline (separate concern from the
ChromaDB long-term semantic memory in memory/vector_store.py -- that
one persists *content* across topics/sessions; this one persists
*pipeline progress* within a single run). Swap MemorySaver for
langgraph.checkpoint.sqlite.SqliteSaver if you want that resumability
to survive a process restart too.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from research_agent_system.config.config import Config
from research_agent_system.workflows.graph_state import ResearchState
from research_agent_system.workflows.nodes import (
    planner_node,
    make_research_node,
    make_summarizer_node,
    synthesis_node,
    critic_node,
    formatter_node,
    route_after_critic,
)


def build_graph(cfg: Config | None = None, with_checkpointer: bool = True):
    cfg = cfg or Config()

    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("research", make_research_node(cfg))
    graph.add_node("summarizer", make_summarizer_node(cfg))
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic", critic_node)
    graph.add_node("formatter", formatter_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "summarizer")
    graph.add_edge("summarizer", "synthesis")
    graph.add_edge("synthesis", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"synthesis": "synthesis", "formatter": "formatter"},
    )

    graph.add_edge("formatter", END)

    checkpointer = MemorySaver() if with_checkpointer else None
    return graph.compile(checkpointer=checkpointer)
