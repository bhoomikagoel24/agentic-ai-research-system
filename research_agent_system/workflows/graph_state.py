"""
research_agent_system/workflows/graph_state.py

V2 replacement for schemas.PipelineState as the thing that flows
through the pipeline. LangGraph wants a TypedDict (or pydantic model)
for StateGraph, with each node returning a partial dict that gets
merged into the running state.

This deliberately mirrors PipelineState's fields 1:1 — same names,
same types — so nothing about how downstream code reads the result
(state["papers"], state["synthesis"], ...) needs to change vs. v1's
state.papers / state.synthesis attribute access. Only `.` became `[]`.
"""

from __future__ import annotations

from typing import TypedDict

from research_agent_system.schemas import (
    ResearchPlan,
    PaperSummary,
    SynthesisOutput,
)


class ResearchState(TypedDict, total=False):
    topic: str

    plan: ResearchPlan | None
    papers: list[dict]
    summaries: list[PaperSummary]
    synthesis: SynthesisOutput | None
    critique: dict
    critique_retries: int
    final_report: str

    metrics: dict
    memory_hits: dict
