"""
research_agent_system/workflows/langgraph_workflow.py

    START -> planner -> research -> summarizer -> synthesis -> critic
                                                       ^             |
                                                       |             v
                                                       +---(loop)--- |
                                                                     v
                                                                 formatter -> END

The critic -> synthesis edge is conditional (route_after_critic):
loops back if the critique found weaknesses, up to
nodes.MAX_CRITIC_REFINEMENTS times, otherwise proceeds to formatter.

AblationConfig controls which components are active — this is what
makes the benchmark ablation studies actually different from each other.
Without this, all configurations would run the same pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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


@dataclass
class AblationConfig:
    """
    Controls which pipeline components are active.
    Pass this to build_graph to create a specific ablation variant.

    disable_critic  → synthesis goes directly to formatter, no loop
    disable_memory  → ChromaDB lookups are skipped, every paper re-summarised
    sources         → which retrieval tools the research agent may call
                      ("arxiv", "semantic_scholar"); omit one to isolate it
    """
    disable_critic: bool = False
    disable_memory: bool = False
    sources: list[str] = field(default_factory=lambda: ["arxiv", "semantic_scholar"])

    @classmethod
    def full(cls) -> "AblationConfig":
        """Complete ARIA — all components active."""
        return cls()

    @classmethod
    def no_critic(cls) -> "AblationConfig":
        """Ablation: remove critique-refinement loop."""
        return cls(disable_critic=True)

    @classmethod
    def no_memory(cls) -> "AblationConfig":
        """Ablation: disable ChromaDB semantic memory."""
        return cls(disable_memory=True)

    @classmethod
    def single_source(cls) -> "AblationConfig":
        """Ablation: arXiv only — removes Semantic Scholar."""
        return cls(sources=["arxiv"])


def build_graph(
    cfg: Config | None = None,
    ablation: AblationConfig | None = None,
    with_checkpointer: bool = True,
):
    """
    Build and compile the ARIA StateGraph.

    Args:
        cfg:              Pipeline configuration (topk, dev_mode, etc.)
        ablation:         Which components to enable/disable. Defaults to
                          full system (all components active).
        with_checkpointer: Attach a MemorySaver checkpointer for crash-resume.

    The ablation parameter is what makes benchmark configurations
    genuinely different — without it, all ablation runs would execute
    the exact same code and produce identical results.
    """
    cfg = cfg or Config()
    ablation = ablation or AblationConfig.full()

    graph = StateGraph(ResearchState)

    # ── nodes ────────────────────────────────────────────────────
    graph.add_node("planner", planner_node)
    graph.add_node("research", make_research_node(cfg, ablation=ablation))
    graph.add_node("summarizer", make_summarizer_node(cfg, ablation=ablation))
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("formatter", formatter_node)

    if not ablation.disable_critic:
        graph.add_node("critic", critic_node)

    # ── edges ────────────────────────────────────────────────────
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "summarizer")
    graph.add_edge("summarizer", "synthesis")

    if ablation.disable_critic:
        # No critic node — synthesis flows directly to formatter.
        # This is the actual difference in the no_critic ablation;
        # without this branch, no_critic would still run the critic.
        graph.add_edge("synthesis", "formatter")
    else:
        graph.add_edge("synthesis", "critic")
        graph.add_conditional_edges(
            "critic",
            route_after_critic,
            {"synthesis": "synthesis", "formatter": "formatter"},
        )

    graph.add_edge("formatter", END)

    checkpointer = MemorySaver() if with_checkpointer else None
    return graph.compile(checkpointer=checkpointer)

