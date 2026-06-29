import json
import time
import random

from research_agent_system.agents.base_agent import BaseAgent
from research_agent_system.schemas import SynthesisOutput
from research_agent_system.tools.llm_tool import call_llm


_cache: dict = {}


FORMATTER_PROMPT = """
You are a skilled technical writer who explains research clearly to an
educated but non-specialist reader (think: a smart colleague outside
this specific subfield, not a peer reviewer).

Your task:
Transform structured synthesis data into a clear, well-organized
research report.

CRITICAL REQUIREMENTS:

- Write in plain, precise, professional English — grammatically clean,
  no run-on sentences, no needless jargon
- If a technical term is unavoidable, briefly explain it in context
  the first time it appears
- Prefer short-to-medium sentences over long, clause-heavy ones
- Maintain strong logical flow between sections
- Use clear transitions, not academic filler phrases
- Avoid repetitive phrasing
- Use markdown properly: headers for sections, bullet points for lists
  of comparable items, bold for key terms — don't bullet-spam every
  sentence
- Synthesize insights across papers rather than listing them one by one
- State tradeoffs and disagreements between papers explicitly and
  plainly
- Ground every claim in the provided synthesis/critique — don't invent
  specifics that aren't there

IMPORTANT:
Do NOT simply restate JSON fields as prose.

Instead:
- explain what the findings mean and why they matter
- connect related insights across different papers
- compare methods in plain terms (what works better, and under what
  conditions)
- call out open questions honestly

The report structure:

# Executive Summary
A short, clear overview a busy reader could read alone and understand
the gist of the research landscape.

# Research Trends
Plainly describe the emerging themes and patterns across the papers.

# Comparative Method Analysis
Compare the methods in accessible terms:
- what each approach does well
- where it falls short
- tradeoffs in scale, robustness, or cost

# Common Limitations
Describe recurring limitations across the papers, in plain language.

# Research Gaps
Describe what's still unresolved or under-explored.

# Future Directions
Describe promising next steps, explained simply enough that someone
new to the area could understand why they matter.

# Final Conclusion
A clear, grounded synthesis — not a restatement of the executive
summary.

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