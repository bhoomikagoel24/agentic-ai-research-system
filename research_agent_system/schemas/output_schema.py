from typing import Literal

from pydantic import (
    BaseModel,
    Field
)


# =====================================================
# RESEARCH PLAN
# =====================================================

class ResearchPlan(BaseModel):

    sub_questions: list[str] = Field(
        min_length=3
    )

    search_queries: list[str] = Field(
        min_length=3
    )


# =====================================================
# PAPER SUMMARY
# =====================================================

class PaperSummary(BaseModel):

    title: str = ""
    problem: str = ""
    method: str = ""
    key_findings: str = ""
    limitations: str = ""
    core_tradeoff: str = ""

    confidence: Literal[
        "high",
        "medium",
        "low"
    ] = "medium"

    quality_score: int = Field(
        default=5,
        ge=1,
        le=10
    )

    source_title: str = ""
    source_url: str = ""
    source_year: int | None = None


# =====================================================
# SYNTHESIS OUTPUT
# =====================================================

class SynthesisOutput(BaseModel):

    common_methods: list[dict] = Field(
        default_factory=list
    )

    method_comparisons: list[dict] = Field(
        default_factory=list
    )

    agreements: list[dict] = Field(
        default_factory=list
    )

    contradictions: list[dict] = Field(
        default_factory=list
    )

    common_limitations: list[dict] = Field(
        default_factory=list
    )

    research_gaps: list[dict] = Field(
        default_factory=list
    )

    emerging_trends: list[dict] = Field(
        default_factory=list
    )

    future_directions: list[dict] = Field(
        default_factory=list
    )

    final_insight: str = ""
    confidence_reasoning: str = ""

    confidence: Literal[
        "high",
        "medium",
        "low"
    ] = "medium"


# =====================================================
# PIPELINE STATE
# =====================================================

class PipelineState(BaseModel):

    topic: str
    plan: ResearchPlan | None = None

    papers: list[dict] = Field(
        default_factory=list
    )

    summaries: list[PaperSummary] = Field(
        default_factory=list
    )

    synthesis: SynthesisOutput | None = None

    critique: dict = Field(
        default_factory=dict
    )

    final_report: str = ""