import time
import random
from research_agent_system.agents.base_agent import BaseAgent
from research_agent_system.schemas import ResearchPlan
from research_agent_system.tools.search_utils import (
    multi_source_fetch,
    filter_papers,
    rank_papers,
    remove_similar_titles,
)
from research_agent_system.memory.memory_manager import (
    MemoryManager
)

_query_cache: dict = {}


def _expand_queries(
    queries: list[str],
    attempt: int
) -> list[str]:

    suffixes = [
        "survey OR review",
        "benchmark dataset comparison",
        "clinical application",
        "limitations challenges",
        "recent advances",
    ]

    return [
        q + " " + suffixes[
            attempt % len(suffixes)
        ]
        for q in queries
    ]


class ResearchAgent(BaseAgent):

    def run(
        self,
        plan: ResearchPlan
    ) -> dict:

        queries = plan.search_queries

        all_papers = []

        attempts = 0

        memory = MemoryManager()

        while attempts < self.cfg.max_attempts:

            self.logger.info(
                f"Research attempt {attempts+1}"
            )

            for q in queries:

                self.logger.info(
                    f"Searching: {q}"
                )

                # ======================================
                # MEMORY SEARCH
                # ======================================

                try:

                    memory_results = memory.search_memory(
                        q,
                        top_k=3
                    )

                    reused_titles = set()

                    for result in memory_results:

                        similarity = result.get(
                            "similarity",
                            0
                        )

                        if similarity > 0.90:

                            self.logger.info(
                                f"Memory reused "
                                f"(sim={similarity:.2f})"
                            )

                            try:

                                import json

                                data = json.loads(
                                    result["document"]
                                )

                                # ==================================
                                # CASE 1:
                                # FULL PAPER OBJECT
                                # ==================================

                                if (
                                    isinstance(data, dict)
                                    and "title" in data
                                    and "abstract" in data
                                ):

                                    title_key = data[
                                        "title"
                                    ].lower().strip()

                                    if title_key not in reused_titles:

                                        reused_titles.add(
                                            title_key
                                        )

                                        all_papers.append(
                                            data
                                        )

                                # ==================================
                                # CASE 2:
                                # SUMMARY OBJECT
                                # ==================================

                                elif (
                                    isinstance(data, dict)
                                    and "source_title" in data
                                ):

                                    self.logger.info(
                                        "Summary memory found "
                                        "- skipping paper reuse"
                                    )

                            except Exception as e:

                                self.logger.warning(
                                    f"Memory parse failed: {e}"
                                )

                except Exception as e:

                    self.logger.warning(
                        f"Memory search failed: {e}"
                    )


                # ======================================
                # API SEARCH
                # ======================================

                try:

                    if q in _query_cache:

                        results = _query_cache[q]

                    else:

                        results = multi_source_fetch(q)

                        _query_cache[q] = results

                    if results:

                        all_papers.extend(results)

                except Exception as e:

                    self.logger.error(
                        f"Search error: {e}"
                    )

                time.sleep(
                    random.uniform(2, 5)
                )

            # ======================================
            # PREVIEW FILTER
            # ======================================

            unique_preview = {}

            for p in all_papers:

                key = (
                    p.get("url")
                    or p["title"].lower().strip()
                )

                if key not in unique_preview:

                    unique_preview[key] = p

            filtered = filter_papers(
                list(unique_preview.values()),
                queries
            )

            filtered = remove_similar_titles(
                filtered
            )

            self.logger.info(
                f"Quality papers found: "
                f"{len(filtered)}"
            )

            if len(filtered) >= self.cfg.min_papers:

                self.logger.info(
                    "Enough papers collected"
                )

                break

            queries = _expand_queries(
                queries,
                attempts + 1
            )

            attempts += 1

        # ======================================
        # FINAL CLEANUP
        # ======================================

        unique = {}

        for p in all_papers:

            key = (
                p.get("url")
                or p["title"].lower().strip()
            )

            if key not in unique:

                unique[key] = p

        papers = list(unique.values())

        papers = filter_papers(
            papers,
            queries
        )

        papers = remove_similar_titles(
            papers
        )

        papers = rank_papers(
            papers
        )

        papers = papers[:self.cfg.topk]

        return {
            "queries_used": queries,
            "total_papers": len(papers),
            "papers": papers
        }