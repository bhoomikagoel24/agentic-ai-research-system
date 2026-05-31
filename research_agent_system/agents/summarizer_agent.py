import os
import json
import time
import random
import hashlib

from research_agent_system.agents.base_agent import BaseAgent
from research_agent_system.schemas import PaperSummary
from research_agent_system.tools.llm_tool import call_llm
from research_agent_system.utils.validators import extract_json


SUMMARY_PROMPT = """
You are a research paper analysis agent.

Analyze the paper deeply.

Return ONLY valid JSON.

Format:
{{
  "title": "",
  "problem": "",
  "method": "",
  "key_findings": "",
  "limitations": "",
  "core_tradeoff": "",
  "confidence": "high/medium/low",
  "quality_score": 1
}}

Rules:
- Be analytical, not descriptive
- Avoid generic summaries
- confidence depends on clarity and relevance
- quality_score reflects summary quality (1-10)

Paper:
Title: {title}

Abstract:
{abstract}
"""


class SummarizerAgent(BaseAgent):

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

                    print(raw)

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