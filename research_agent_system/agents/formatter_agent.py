import json
import time
import random

from research_agent_system.agents.base_agent import BaseAgent
from research_agent_system.schemas import SynthesisOutput
from research_agent_system.tools.llm_tool import call_llm


_cache: dict = {}


FORMATTER_PROMPT = """
You are an elite academic literature review writer.

Your task:
Transform structured synthesis data into a highly polished,
publication-style literature review report.

CRITICAL REQUIREMENTS:

- Write like a top-tier research analyst
- Maintain strong logical flow between sections
- Use analytical transitions
- Avoid repetitive phrasing
- Avoid bullet spam
- Synthesize insights across papers
- Highlight tradeoffs explicitly
- Discuss methodological evolution
- Mention conflicting findings carefully
- Make arguments evidence-grounded
- Use formal academic tone

IMPORTANT:
Do NOT simply restate JSON fields.

Instead:
- infer relationships
- connect insights
- explain implications
- compare methodologies
- discuss why trends matter

The report structure:

# Executive Summary
High-level overview of major findings and overall research landscape.

# Research Trends
Discuss emerging themes and patterns across studies.

# Comparative Method Analysis
Compare methods deeply:
- strengths
- weaknesses
- tradeoffs
- scalability
- robustness

# Common Limitations
Analyze recurring technical and methodological limitations.

# Research Gaps
Discuss unresolved problems and underexplored areas.

# Future Directions
Explain promising future opportunities and next-generation approaches.

# Final Conclusion
Provide a strong synthesis-driven conclusion.

CRITIC FEEDBACK:
{critic_feedback}

SYNTHESIS:
{synthesis}
"""

class FormatterAgent(BaseAgent):

    def run(
        self,
        synthesis: SynthesisOutput,
        critique: dict = None
    ) -> str:

        self.logger.info(
            "Running formatter agent"
        )

        text = json.dumps(
            synthesis.model_dump()
            if hasattr(synthesis, "model_dump")
            else synthesis,
            indent=2,
            ensure_ascii=False
        )

        # ==============================
        # CACHE
        # ==============================

        if text in _cache:

            self.logger.info(
                "Loaded report from cache"
            )

            return _cache[text]

        critic_text = json.dumps(
            critique,
            indent=2,
            ensure_ascii=False
        ) if critique else "No critique available"

        prompt = FORMATTER_PROMPT.format(
            synthesis=text,
            critic_feedback=critic_text
        )

        for attempt in range(3):

            try:

                report = call_llm(prompt)

                if (
                    report and
                    len(report.strip()) > 100
                ):

                    _cache[text] = report

                    return report

            except Exception as e:

                self.logger.warning(
                    f"Formatter retry "
                    f"{attempt+1}: {e}"
                )

                time.sleep(
                    (2 ** attempt)
                    + random.uniform(1, 3)
                )

        self.logger.error(
            "Formatting failed"
        )

        return "Formatting failed."