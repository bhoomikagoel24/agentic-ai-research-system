import time
import random
from research_agent_system.agents.base_agent import BaseAgent
from research_agent_system.config.config import Config
from research_agent_system.schemas import ResearchPlan
from research_agent_system.tools.search_utils import (
    multi_source_fetch, filter_papers,
    rank_papers, remove_similar_titles,
)
from research_agent_system.utils.logger import get_logger

# logger = get_logger(__name__)
_query_cache: dict = {}


def _expand_queries(queries: list[str], attempt: int) -> list[str]:
    suffixes = [
        "survey OR review",
        "benchmark dataset comparison",
        "clinical application",
        "limitations challenges",
        "recent advances",
    ]
    return [q + " " + suffixes[attempt % len(suffixes)] for q in queries]


class ResearchAgent(BaseAgent):
    def run(self, plan: ResearchPlan) -> dict:
        queries    = plan.search_queries
        all_papers = []
        attempts   = 0

        while attempts < self.cfg.max_attempts:
            self.logger.info(f"Research attempt {attempts+1}")
            for q in queries:
                self.logger.info(f"Searching: {q}")
                try:
                    if q in _query_cache:
                        results = _query_cache[q]
                    else:
                        results = multi_source_fetch(q)
                        _query_cache[q] = results
                    if results:
                        all_papers.extend(results)
                except Exception as e:
                    self.logger.error(f"Search error: {e}")
                time.sleep(random.uniform(4, 7))

            # preview dedup + filter
            unique_preview = {}
            for p in all_papers:
                key = p.get("url") or p["title"].lower().strip()
                if key not in unique_preview:
                    unique_preview[key] = p

            filtered = filter_papers(list(unique_preview.values()), queries)
            filtered = remove_similar_titles(filtered)
            self.logger.info(f"Quality papers found: {len(filtered)}")

            if len(filtered) >= self.cfg.min_papers:
                self.logger.info("Enough papers collected")
                break

            queries = _expand_queries(queries, attempts + 1)
            attempts += 1

        # final pipeline
        unique = {}
        for p in all_papers:
            key = p.get("url") or p["title"].lower().strip()
            if key not in unique:
                unique[key] = p

        papers = list(unique.values())
        papers = filter_papers(papers, queries)
        papers = remove_similar_titles(papers)
        papers = rank_papers(papers)
        papers = papers[:self.cfg.topk]

        return {"queries_used": queries, "total_papers": len(papers), "papers": papers}