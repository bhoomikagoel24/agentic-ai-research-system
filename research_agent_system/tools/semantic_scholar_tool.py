import time
import random
import requests

from research_agent_system.utils.logger import (
    get_logger
)


logger = get_logger(__name__)

_cache: dict = {}


def fetch_from_semantic_scholar(
    query: str,
    limit: int = 2
) -> list[dict]:

    key = f"{query}_{limit}"

    # ========================================
    # CACHE
    # ========================================

    if key in _cache:

        logger.info(
            "Semantic Scholar: cache hit"
        )

        return _cache[key]

    url = (
        "https://api.semanticscholar.org/"
        "graph/v1/paper/search"
    )

    params = {
        "query": query,
        "limit": limit,
        "fields":
            "title,abstract,"
            "year,authors,url"
    }

    headers = {
        "User-Agent":
            "ResearchSynthesisAgent/1.0"
    }

    for attempt in range(3):

        try:

            r = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=10
            )

            # ====================================
            # RATE LIMIT
            # ====================================

            if r.status_code == 429:

                logger.warning(
                    "Semantic Scholar "
                    "rate limited"
                )
                time.sleep((2 ** attempt)+ random.uniform(2, 5))

                continue

            r.raise_for_status()

            papers = []

            for p in r.json().get(
                "data",
                []
            ):

                if not p.get("title"):
                    continue

                papers.append({

                    "title":p["title"],

                    "abstract":
                        p.get(
                            "abstract",
                            ""
                        ),

                    "year":p.get("year"),

                    "url":p.get("url"),

                    "authors": [
                        a.get("name", "")
                        for a in p.get(
                            "authors",
                            []
                        )
                    ],

                    "source":
                        "semantic_scholar",
                })

            _cache[key] = papers

            time.sleep(random.uniform(2, 4))

            return papers

        except Exception as e:

            logger.warning(
                f"Semantic retry "
                f"{attempt+1}: {e}"
            )

            time.sleep((2 ** attempt) + random.uniform(1, 3))

    logger.error("Semantic Scholar failed")

    return []