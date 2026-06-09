import json
import sys
from research_agent_system.config.config import Config
from research_agent_system.schemas import PipelineState, SynthesisOutput, PaperSummary
from research_agent_system.agents import (
    PlannerAgent, ResearchAgent, SummarizerAgent,
    SynthesisAgent, CriticAgent, FormatterAgent,
)
from research_agent_system.agents.synthesis_agent import evaluate_synthesis
from research_agent_system.tools.llm_tool import call_llm
from research_agent_system.utils.validators import extract_json
from research_agent_system.utils.logger import get_logger
from research_agent_system.utils.exception import ResearchAgentException

from research_agent_system.memory.memory_manager import (
    MemoryManager
)

logger = get_logger(__name__)


def run_pipeline(
    topic: str,
    cfg: Config = None,
    enable_critic: bool = True,
    enable_formatter: bool = True,
    on_step: callable = None,
) -> PipelineState:
    
    if cfg is None:
        cfg = Config()
    
    memory = MemoryManager()
    memory_results = memory.search_memory(
        topic,
        top_k=3
    )

    reused_summaries = []

    for result in memory_results:

        similarity = result.get(
            "similarity",
            0
        )

        logger.info(
            f"\nSimilarity: "
            f"{similarity:.3f}"
        )

        # ==================================
        # REUSE ONLY HIGH SIMILARITY
        # ==================================

        if similarity > 0.85:

            try:

                data = json.loads(
                    result["document"]
                )

                reused_summaries.append(
                    PaperSummary(**data)
                )

                logger.info(
                    "\nREUSED MEMORY SUMMARY\n"
                )

                logger.info(
                    data.get(
                        "source_title",
                        "Unknown"
                    )
                )

            except Exception as e:

                logger.info(
                    f"Memory parse failed: {e}"
                )

        else:

            logger.info(
                "\nLOW SIMILARITY → "
                "REGENERATE\n"
            )

    logger.info("\n========== MEMORY RESULTS ==========\n")

    logger.info(memory_results)

    logger.info("\n====================================\n")

    def step(name: str, msg: str):
        logger.info(msg)
        if on_step:
            on_step(name, msg)

    state = PipelineState(topic=topic)

    # ── Step 1: Plan ─────────────────────────────────────────
    # step("planner", "Planning research queries...")
    step("planner", "running")
    try:
        state.plan = PlannerAgent(cfg).run(topic)
        step("planner", "done")
        logger.info(f"Plan ready — {len(state.plan.search_queries)} queries")
    except ResearchAgentException as e:
        logger.error(f"Pipeline aborted at planning: {e}")
        raise

    # ── Step 2: Research ──────────────────────────────────────
    step("research", "running")
    out = ResearchAgent(cfg).run(state.plan)
    state.papers = out["papers"]

    # ======================================
    # SAVE PAPERS TO MEMORY
    # ======================================

    for paper in state.papers:

        try:

            memory.save_paper(paper)

        except Exception as e:

            logger.warning(
                f"Paper memory save failed: {e}"
            )

    step("research", "done")
    logger.info(f"Papers retrieved: {len(state.papers)}")

    with open("data/raw_papers.json", "w", encoding="utf-8") as f:
        json.dump(
            state.papers,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ── Step 3: Summarize ─────────────────────────────────────
    step("summarizer", "running")
    # state.summaries = SummarizerAgent(cfg).run(state.papers)
    papers_to_summarize = []

    reused_titles = {
        s.source_title.lower()
        for s in reused_summaries
    }

    for p in state.papers:

        if p["title"].lower() not in reused_titles:

            papers_to_summarize.append(p)

    new_summaries = SummarizerAgent(cfg).run(
        papers_to_summarize
    )

    state.summaries = (
        reused_summaries +
        new_summaries
    )

    for summary in new_summaries:

        try:

            matching_paper = next(
                (
                    p for p in state.papers
                    if p["title"].lower()
                    == summary.source_title.lower()
                ),
                None
            )

            if matching_paper:

                memory.save_summary(
                    matching_paper,
                    summary.model_dump()
                )

        except Exception as e:

            logger.warning(
                f"Memory save failed: {e}"
            )
    # =====================================
    # REMOVE DUPLICATE SUMMARIES
    # =====================================
    unique = {}

    for s in state.summaries:

        key = s.source_title.lower()

        if key not in unique:
            unique[key] = s

    state.summaries = list(
        unique.values()
    )

    step("summarizer", "done")
    logger.info(f"Summaries done: {len(state.summaries)}")

    # ── Step 4: Pipeline metrics ──────────────────────────────
    metrics = evaluate_pipeline(state)
    logger.info(f"Pipeline metrics: {json.dumps(metrics)}")

    # ── Step 5: Synthesize ────────────────────────────────────
    step("synthesis", "running")
    try:
        state.synthesis = SynthesisAgent(cfg).run(state.summaries)
        step("synthesis","done")
        synth_metrics = evaluate_synthesis(state.synthesis)
        logger.info(f"Synthesis metrics: {json.dumps(synth_metrics)}")
    except ResearchAgentException as e:
        logger.error(f"Pipeline aborted at synthesis: {e}")
        raise

    # ── Step 6: Critique + Refinement ────────────────────────
    if enable_critic:
        step("critic", "running")
        state.critique = CriticAgent(cfg).run(state.synthesis)
        step("critic", "done")
        weaknesses = (
            state.critique.get(
                "major_weaknesses",
                []
            )
            if state.critique
            else []
        )
        if weaknesses:
            step("critic", f"Refining synthesis ({len(weaknesses)} weaknesses)...")
            try:
                refinement_prompt = f"""
                Improve this research synthesis JSON.
                Address these weaknesses: {json.dumps(weaknesses[:2])}

                Return ONLY valid JSON in exact same format.
                Every field that is a list must contain dicts, not strings.

                Original:
                {
                    json.dumps(
                        state.synthesis.model_dump(),
                        indent=2,
                        ensure_ascii=False
                    )[:1200]
                }
                """
                raw  = call_llm(refinement_prompt)
                data = extract_json(raw)

                list_fields = [
                    "common_methods", "method_comparisons", "agreements",
                    "contradictions", "common_limitations", "research_gaps",
                    "emerging_trends", "future_directions"
                ]
                valid = all(
                    all(isinstance(x, dict) for x in data.get(f, []))
                    for f in list_fields
                )
                if valid:
                    state.synthesis = SynthesisOutput(**data)
                    logger.info("Synthesis refined successfully")
                else:
                    logger.warning("Refinement had wrong types — keeping original")

            except Exception as e:
                logger.warning(f"Refinement failed, keeping original: {e}")

    # ── Step 7: Format ────────────────────────────────────────
    if enable_formatter:
        step("formatter", "running")
        # state.final_report = FormatterAgent(cfg).run(state.synthesis)
        state.final_report = FormatterAgent(cfg).run(
            state.synthesis,
            state.critique
        )
        step("formatter", "done")
        with open(
            "data/final_report.md",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(state.final_report)

    # ── Save state ────────────────────────────────────────────
    with open(
        "state.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            state.model_dump(),
            f,
            indent=2,
            default=str,
            ensure_ascii=False
        )

    logger.info("Pipeline complete")
    return state


def evaluate_pipeline(state: PipelineState) -> dict:
    papers    = state.papers
    summaries = state.summaries

    return {
        "total_papers": len(papers),
        "total_summaries": len(summaries),
        "avg_relevance_score": round(
            sum(p.get("relevance_score", 0) for p in papers)
            / max(len(papers), 1), 3
        ),
        "avg_summary_quality": round(
            sum(s.quality_score for s in summaries)
            / max(len(summaries), 1), 2
        ),
        "high_confidence_summaries": sum(
            1 for s in summaries if s.confidence == "high"
        ),
    }