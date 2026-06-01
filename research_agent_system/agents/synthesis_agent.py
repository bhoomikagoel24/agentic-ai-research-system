import sys
import json

from research_agent_system.agents.base_agent import BaseAgent

from research_agent_system.schemas import (
    PaperSummary,
    SynthesisOutput
)

from research_agent_system.tools.llm_tool import (
    call_llm
)

from research_agent_system.utils.validators import (
    extract_json
)

from research_agent_system.utils.exception import (
    ResearchAgentException
)


SYNTHESIS_PROMPT = """
You are an advanced AI research synthesis agent.

Analyze multiple research paper summaries.

Your job is to reason ACROSS papers,
not summarize individually.

Return ONLY valid JSON.

CRITICAL:
- No markdown
- No explanation
- No empty lists

Format:
{{
  "common_methods": [
    {{
      "method": "",
      "source_papers": []
    }}
  ],

  "method_comparisons": [
    {{
      "comparison": "",
      "source_papers": [],
      "evidence_reasoning": "",
      "core_tradeoff": ""
    }}
  ],

  "agreements": [
    {{
      "insight": "",
      "supporting_papers": [],
      "why_supported": ""
    }}
  ],

  "contradictions": [
    {{
      "issue": "",
      "conflicting_papers": [],
      "reason_for_conflict": ""
    }}
  ],

  "common_limitations": [
    {{
      "limitation": "",
      "why_it_occurs": "",
      "affected_methods": []
    }}
  ],

  "research_gaps": [
    {{
      "gap": "",
      "why_important": ""
    }}
  ],

  "emerging_trends": [
    {{
      "trend": "",
      "evidence_sources": [],
      "why_emerging": ""
    }}
  ],

  "future_directions": [
    {{
      "direction": "",
      "motivation": ""
    }}
  ],

  "final_insight": "",

  "confidence_reasoning": "",

  "confidence": "high"
}}

Research Summaries:
{summaries}
"""


class SynthesisAgent(BaseAgent):

    def run(
        self,
        summaries: list[PaperSummary]
    ) -> SynthesisOutput:

        self.logger.info(
            "Running synthesis agent"
        )

        compressed = []

        for s in summaries:

            d = (
                s.model_dump()
                if hasattr(s, "model_dump")
                else s
            )

            compressed.append({

                "title":
                    d.get(
                        "source_title",
                        ""
                    )[:80],

                "problem":
                    d.get(
                        "problem",
                        ""
                    )[:120],

                "method":
                    d.get(
                        "method",
                        ""
                    )[:120],

                "key_findings":
                    d.get(
                        "key_findings",
                        ""
                    )[:150],

                "limitations":
                    d.get(
                        "limitations",
                        ""
                    )[:100],

                "core_tradeoff":
                    d.get(
                        "core_tradeoff",
                        ""
                    )[:80],

                "quality_score":
                    d.get(
                        "quality_score"
                    ),

                "confidence":
                    d.get(
                        "confidence"
                    ),
            })

        prompt = SYNTHESIS_PROMPT.format(
            summaries=json.dumps(
                compressed,
                indent=2,
                ensure_ascii=False
            )
        )

        try:

            raw = call_llm(prompt)

            print(
                "\n\n========== RAW SYNTHESIS ==========\n"
            )

            print(raw)

            print(
                "\n===================================\n"
            )

            if not raw:

                raise ValueError(
                    "Empty synthesis response"
                )

            raw = raw.strip()

            data = extract_json(raw)

            return SynthesisOutput(
                **data
            )

        except Exception as e:

            self.logger.error(
                f"Synthesis error: {e}"
            )

            raise ResearchAgentException(
                f"Synthesis failed: {e}",
                sys
            )


# ============================================
# EVALUATION
# ============================================

def evaluate_synthesis(
    synthesis: SynthesisOutput
) -> dict:

    return {

        "num_comparisons":
            len(
                synthesis.method_comparisons
            ),

        "num_gaps":
            len(
                synthesis.research_gaps
            ),

        "num_trends":
            len(
                synthesis.emerging_trends
            ),

        "has_method_comparisons":
            len(
                synthesis.method_comparisons
            ) > 0,
    }