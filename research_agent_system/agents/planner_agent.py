import json
import time
import sys

from pydantic import ValidationError

from research_agent_system.agents.base_agent import BaseAgent
from research_agent_system.schemas import ResearchPlan
from research_agent_system.tools.llm_tool import call_llm
from research_agent_system.utils.validators import extract_json
from research_agent_system.utils.exception import (
    ResearchAgentException
)

from research_agent_system.memory.memory_manager import (
    MemoryManager
)

PLANNER_PROMPT = """
You are a research planning agent.

Break the given topic into structured
research steps.

Return ONLY valid JSON.

No markdown.
No explanation.

Format:
{{
  "sub_questions": [
    "...",
    "...",
    "..."
  ],

  "search_queries": [
    "...",
    "...",
    "..."
  ]
}}

Rules:
- Each list must have 3-6 items
- Avoid generic queries
- Make queries research-grade

Topic:
{topic}
"""


EVAL_PROMPT = """
You are a research query evaluator.

Evaluate the quality of generated
research queries.

Return ONLY valid JSON.

Format:
{{
  "is_valid": true,

  "issues": [
    "..."
  ],

  "improved_queries": [
    "...",
    "...",
    "..."
  ]
}}

Rules:
- If queries are good:
  is_valid = true

- If queries are weak:
  improve them

- Avoid duplicates
- Avoid generic phrasing

Queries:
{queries}
"""


class PlannerAgent(BaseAgent):

    def run(
        self,
        topic: str
    ) -> ResearchPlan:
        
        memory = MemoryManager()

        memory_results = memory.search_memory(
            topic,
            top_k=1
        )

        for result in memory_results:

            similarity = result.get(
                "similarity",
                0
            )

            if similarity > 0.92:

                try:

                    data = json.loads(
                        result["document"]
                    )

                    if (
                        isinstance(data, dict)
                        and "search_queries" in data
                        and "sub_questions" in data
                    ):

                        self.logger.info(
                            "Reusing planner memory"
                        )

                        return ResearchPlan(**data)

                except Exception:
                    pass

        prompt = PLANNER_PROMPT.format(
            topic=topic
        )

        for i in range(3):

            self.logger.info(
                f"Planner attempt {i+1}"
            )

            try:

                raw = call_llm(prompt)

                print(
                    "\n\n========== RAW PLANNER ==========\n"
                )

                print(raw)

                print(
                    "\n=================================\n"
                )

                plan = extract_json(
                    raw,
                    model=ResearchPlan
                )

                queries = plan.search_queries

                # =====================================
                # QUERY EVALUATION LOOP
                # =====================================

                for j in range(3):

                    self.logger.info(
                        f"Eval cycle {j+1}"
                    )

                    try:

                        eval_prompt = EVAL_PROMPT.format(
                            queries=json.dumps(
                                queries,
                                indent=2
                            )
                        )

                        eval_raw = call_llm(
                            eval_prompt
                        )

                        print(
                            "\n\n========== RAW EVAL ==========\n"
                        )

                        print(eval_raw)

                        print(
                            "\n==============================\n"
                        )

                        eval_data = extract_json(
                            eval_raw
                        )

                    except Exception as e:

                        self.logger.warning(
                            f"Eval failed: {e}"
                        )

                        time.sleep(2 ** j)

                        continue

                    # =================================
                    # VALIDATED
                    # =================================

                    if eval_data.get("is_valid"):

                        self.logger.info(
                            "Planner finalized"
                        )

                        plan_data = {
                            "sub_questions":
                                plan.sub_questions,

                            "search_queries":
                                queries
                        }

                        memory.save_plan(
                            topic,
                            plan_data
                        )

                        return ResearchPlan(
                            sub_questions=(
                                plan.sub_questions
                            ),
                            search_queries=queries
                        )

                    # =================================
                    # IMPROVE QUERIES
                    # =================================

                    queries = eval_data.get(
                        "improved_queries",
                        queries
                    )

                return ResearchPlan(
                    sub_questions=(
                        plan.sub_questions
                    ),
                    search_queries=queries
                )

            except (
                ValueError,
                ValidationError
            ) as e:

                self.logger.warning(
                    f"Planner failed: {e}"
                )

                time.sleep(2 ** i)

        plan_data = {
            "sub_questions":
                plan.sub_questions,

            "search_queries":
                queries
        }

        memory.save_plan(
            topic,
            plan_data
        ) 

        raise ResearchAgentException(
            "Planner failed after all retries",
            sys
        )