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

Your job is to reason ACROSS papers — compare them, find where they agree,
where they contradict, what collectively they prove. Do NOT summarize each
paper individually. If you catch yourself writing about "Paper X", stop and
instead say what X and Y together imply.

CRITICAL RULES:
- Every comparison must name which papers are being compared
- key_findings must be quoted or closely paraphrased — do not generalize them
- method_comparisons must state a concrete tradeoff with evidence, not just "method A is different from method B"
- contradictions must explain WHY the conflict exists, not just that it exists
- research_gaps must follow from the limitations actually present, not generic AI research gaps
- No markdown, no explanation outside the JSON
- No empty lists — if a category genuinely doesn't apply, write one entry explaining why

Return ONLY valid JSON.

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
      "comparison": "Paper A uses X while Paper B uses Y — concretely explain the difference and when each is better",
      "source_papers": [],
      "evidence_reasoning": "What in the findings supports this comparison",
      "core_tradeoff": "What does each approach gain and what does it sacrifice"
    }}
  ],

  "agreements": [
    {{
      "insight": "What multiple papers collectively establish as true",
      "supporting_papers": [],
      "why_supported": "What specific finding from each paper supports this"
    }}
  ],

  "contradictions": [
    {{
      "issue": "Where papers reach different conclusions on the same question",
      "conflicting_papers": [],
      "reason_for_conflict": "Why they likely disagree — different datasets, methods, evaluation criteria?"
    }}
  ],

  "common_limitations": [
    {{
      "limitation": "Specific limitation, not a generic one",
      "why_it_occurs": "Structural reason this limitation exists",
      "affected_methods": []
    }}
  ],

  "research_gaps": [
    {{
      "gap": "Specific gap that follows from what these papers collectively did NOT address",
      "why_important": "What would change if this gap were closed"
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
      "direction": "Concrete next research step, not a generic call for 'more research'",
      "motivation": "Which limitation or gap directly motivates this direction"
    }}
  ],

  "final_insight": "One paragraph: what do these papers collectively establish that none of them establishes alone?",

  "confidence_reasoning": "Why you are confident or uncertain about this synthesis",

  "confidence": "high/medium/low"
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
                    )[:120],

                "problem":
                    d.get(
                        "problem",
                        ""
                    )[:250],

                "method":
                    d.get(
                        "method",
                        ""
                    )[:300],

                "key_findings":
                    d.get(
                        "key_findings",
                        ""
                    )[:500],

                "limitations":
                    d.get(
                        "limitations",
                        ""
                    )[:250],

                "core_tradeoff":
                    d.get(
                        "core_tradeoff",
                        ""
                    )[:200],

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