import os
import json
import time
import random
import hashlib

from research_agent_system.agents.base_agent import BaseAgent
from research_agent_system.schemas import PaperSummary
from research_agent_system.tools.llm_tool import call_llm
from research_agent_system.utils.validators import extract_json

from research_agent_system.memory.memory_manager import (
    MemoryManager
)


SUMMARY_PROMPT = """
You are a research paper analyst extracting precise, specific information.

TASK: Analyze this paper and extract concrete details — not definitions, not
generic descriptions. If something is genuinely unclear from the abstract,
say so briefly rather than making a vague claim.

Return ONLY valid JSON, no markdown, no explanation.

Format:
{{
  "title": "",
  "problem": "The specific problem this paper addresses (1-2 sentences, concrete)",
  "method": "The specific method/approach used — name it, describe its mechanism briefly",
  "key_findings": "The actual quantitative or qualitative result — what did they find, measure, or prove? Include numbers if present.",
  "limitations": "What the authors themselves admit, or what is structurally limited by their approach",
  "core_tradeoff": "The central tension in this work — what does the method gain vs. what does it sacrifice?",
  "confidence": "high/medium/low — based on clarity of abstract and methodological rigor described",
  "quality_score": 1
}}

Rules:
- key_findings must state WHAT was found, not just that something was studied
- method must name the actual technique, not just "machine learning" or "deep learning"
- core_tradeoff is mandatory — every method has one, find it
- quality_score: 8-10 = clear findings with numbers, 5-7 = clear but no metrics, 1-4 = vague or too brief to assess
- Do NOT use phrases like "the paper explores" or "the authors investigate" — state the finding directly

Paper:
Title: {title}

Abstract:
{abstract}
"""


class SummarizerAgent(BaseAgent):

    def __init__(self, cfg):

        super().__init__(cfg)
        self.memory = MemoryManager()

    def run(
        self,
        papers: list[dict]
    ) -> list[PaperSummary]:

        summaries: list[PaperSummary] = []

        to_process = (
            papers[:2]
            if self.cfg.dev_mode
            else papers
        )

        for i, p in enumerate(to_process):

            # =====================================
            # STABLE CACHE KEY
            # =====================================

            paper_id = hashlib.md5(
                p["title"].encode("utf-8")
            ).hexdigest()

            cache_file = (
                f"{self.cfg.cache_dir}/{paper_id}.json"
            )

            # =====================================
            # CACHE LOAD
            # =====================================

            if os.path.exists(cache_file):

                with open(
                    cache_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    summaries.append(
                        PaperSummary(**json.load(f))
                    )

                self.logger.info(
                    f"Cache hit: summary {i+1}"
                )

                continue

            # =====================================
            # GENERATE SUMMARY
            # =====================================

            self.logger.info(
                f"Summarizing "
                f"{i+1}/{len(to_process)}: "
                f"{p['title'][:60]}"
            )

            abstract = " ".join(
                p["abstract"].split()[:250]
            )

            prompt = SUMMARY_PROMPT.format(
                title=p["title"],
                abstract=abstract
            )

            success = False

            for attempt in range(2):

                try:

                    raw = call_llm(prompt)

                    # =================================
                    # DEBUG RAW OUTPUT
                    # =================================

                    print(
                        "\n\n========== RAW LLM ==========\n"
                    )

                    self.logger.info(raw)

                    print(
                        "\n=============================\n"
                    )

                    # =================================
                    # JSON EXTRACTION
                    # =================================

                    summary = extract_json(
                        raw,
                        model=PaperSummary
                    )

                    # =================================
                    # QUALITY CHECK
                    # =================================

                    if len(
                        summary.key_findings.strip()
                    ) < 15:

                        self.logger.warning(
                            "Weak findings — retrying"
                        )

                        continue

                    # =================================
                    # METADATA
                    # =================================

                    summary.source_title = p["title"]

                    summary.source_url = p.get(
                        "url",
                        ""
                    )

                    summary.source_year = p.get(
                        "year"
                    )

                    summary.confidence = (
                        "high"
                        if summary.quality_score >= 8
                        else "medium"
                        if summary.quality_score >= 5
                        else "low"
                    )

                    summaries.append(summary)

                    # Memory save 
                    self.memory.save_summary(
                        paper=p,
                        summary=summary.model_dump()
                    )
                    # =================================
                    # CACHE SAVE
                    # =================================

                    with open(
                        cache_file,
                        "w",
                        encoding="utf-8"
                    ) as f:

                        json.dump(
                            summary.model_dump(),
                            f,
                            indent=2,
                            ensure_ascii=False
                        )

                    success = True
                    break

                except Exception as e:

                    self.logger.error(
                        f"Summary attempt "
                        f"{attempt+1}: {e}"
                    )

            # =====================================
            # FAILURE
            # =====================================

            if not success:

                self.logger.warning(
                    f"Skipping: "
                    f"{p['title'][:60]}"
                )

            time.sleep(
                random.uniform(2, 4)
            )

        return summaries