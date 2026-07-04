"""
research_agent_system/schemas/graph_schemas.py

V2-only schemas. Kept in a separate file (not output_schema.py) so
v1's schemas/output_schema.py — and everything that imports from it —
is left completely untouched.

These give structured shape to two things that v1 handled as raw
dicts: the planner's query-evaluation step, and the critic's output.
"""

from typing import Literal

from pydantic import BaseModel, Field


class QueryEvaluation(BaseModel):
    """Mirrors planner_agent.EVAL_PROMPT's JSON contract."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    improved_queries: list[str] = Field(default_factory=list)


class CritiqueOutput(BaseModel):
    """Mirrors critic_agent.CRITIC_PROMPT's JSON contract."""

    reasoning_depth_score: int = Field(default=0, ge=0, le=10)
    grounding_score: int = Field(default=0, ge=0, le=10)
    trend_quality_score: int = Field(default=0, ge=0, le=10)
    gap_analysis_score: int = Field(default=0, ge=0, le=10)

    hallucination_risk: Literal["low", "medium", "high"] = "medium"
    overall_quality: Literal["low", "medium", "high"] = "medium"

    major_weaknesses: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    final_verdict: str = ""
