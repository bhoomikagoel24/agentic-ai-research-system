import json
import time
import random

from research_agent_system.agents.base_agent import BaseAgent
from research_agent_system.schemas import SynthesisOutput
from research_agent_system.tools.llm_tool import call_llm
from research_agent_system.utils.validators import extract_json


_cache: dict = {}


CRITIC_PROMPT = """
You are an expert AI research critic.

Critically evaluate this research synthesis output.

Evaluate on:
1. Depth of reasoning
2. Cross-paper comparison quality
3. Research gap quality
4. Trend analysis quality
5. Evidence grounding
6. Hallucination risk
7. Specificity vs generic statements
8. Overall research usefulness

Be brutally honest.

Detect:
- shallow reasoning
- generic insights
- unsupported claims
- weak comparisons
- repetition

Return ONLY valid JSON.

Format:
{{
  "reasoning_depth_score": 0,
  "grounding_score": 0,
  "trend_quality_score": 0,
  "gap_analysis_score": 0,
  "hallucination_risk": "low/medium/high",
  "overall_quality": "low/medium/high",
  "major_weaknesses": [],
  "improvement_suggestions": [],
  "final_verdict": ""
}}

SYNTHESIS:
{synthesis}
"""


class CriticAgent(BaseAgent):

    def run(
        self,
        synthesis: SynthesisOutput
    ) -> dict:

        self.logger.info(
            "Running critic agent"
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
                "Loaded critique from cache"
            )

            return _cache[text]

        prompt = CRITIC_PROMPT.format(
            synthesis=text
        )

        for attempt in range(3):

            try:

                raw = call_llm(prompt)

                data = extract_json(raw)

                if not data:

                    raise ValueError(
                        "Empty critique output"
                    )

                _cache[text] = data

                return data

            except Exception as e:

                self.logger.warning(
                    f"Critic retry "
                    f"{attempt+1}: {e}"
                )

                time.sleep(
                    (2 ** attempt)
                    + random.uniform(1, 3)
                )

        self.logger.error(
            "Critic failed"
        )

        return {}