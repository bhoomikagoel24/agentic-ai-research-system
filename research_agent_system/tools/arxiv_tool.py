import time
import random
import arxiv
from research_agent_system.utils.logger import get_logger

logger = get_logger(__name__)
_cache: dict = {}


def fetch_from_arxiv(query: str, limit: int = 2) -> list[dict]:
    key = f"{query}_{limit}"
    if key in _cache:
        logger.info("arXiv: cache hit")
        return _cache[key]

    for attempt in range(3):
        try:
            search = arxiv.Search(
                query=query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            client = arxiv.Client(page_size=5, delay_seconds=5)
            papers = []
            for r in client.results(search):
                papers.append({
                    "title":   r.title,
                    "abstract":r.summary,
                    "year":    r.published.year,
                    "url":     r.entry_id,
                    "authors": [a.name for a in r.authors],
                    "source":  "arxiv",
                })
                time.sleep(random.uniform(1, 2))
            _cache[key] = papers
            return papers
        except Exception as e:
            logger.warning(f"arXiv retry {attempt+1}: {e}")
            time.sleep((2 ** attempt) + random.uniform(1, 3))

    logger.error("arXiv failed")
    return []