import time
import random
import os
import requests

from research_agent_system.utils.logger import (
    get_logger
)


logger = get_logger(__name__)

_cache: dict = {}
_call_count: int = 0  # read by benchmark/run_experiments.py APICallTracker

# Free, instant-approval key: https://www.semanticscholar.org/product/api
# Without one, every call shares the public tier's rate limit with the
# entire internet — this is what caused most of the lost time in testing.
# With a key set, the limit is high enough that retries rarely fire at all.
_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()

# Circuit breaker: once this many CONSECUTIVE rate-limit hits land in this
# process, stop calling Semantic Scholar for the rest of this run instead
# of paying ~10-20s of backoff on every remaining query. arXiv still
# covers the search on its own — failing fast beats failing slowly.
_consecutive_rate_limits = 0
_CIRCUIT_BREAKER_THRESHOLD = 2
_circuit_open = False


def fetch_from_semantic_scholar(
    query: str,
    limit: int = 2
) -> list[dict]:

    global _consecutive_rate_limits, _circuit_open

    key = f"{query}_{limit}"

    # ========================================
    # CACHE
    # ========================================

    if key in _cache:

        logger.info(
            "Semantic Scholar: cache hit"
        )

        return _cache[key]

    global _call_count, _consecutive_rate_limits, _circuit_open

    if _circuit_open:
        logger.info(
            "Semantic Scholar: circuit open — skipping"
        )
        return []

    key = f"{query}_{limit}"
    if key in _cache:
        logger.info("Semantic Scholar: cache hit")
        return _cache[key]

    _call_count += 1

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
    if _API_KEY:
        headers["x-api-key"] = _API_KEY

    # Fewer attempts and tighter backoff when there's no key — 3 long
    # exponential backoffs per query was the dominant cost across a
    # 6-query plan with no key set.
    max_attempts = 3 if _API_KEY else 2

    for attempt in range(max_attempts):

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

                _consecutive_rate_limits += 1
                if _consecutive_rate_limits >= _CIRCUIT_BREAKER_THRESHOLD:
                    _circuit_open = True
                    logger.warning(
                        "Semantic Scholar: opening circuit breaker for "
                        "the rest of this run"
                    )
                    return []

                time.sleep((1.5 ** attempt) + random.uniform(1, 2))
                continue

            r.raise_for_status()
            _consecutive_rate_limits = 0  # reset streak on any success

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

            # The unauthenticated tier needs a courtesy delay between
            # calls; an API key's much higher limit doesn't.
            if not _API_KEY:
                time.sleep(random.uniform(1, 2))

            return papers

        except Exception as e:

            logger.warning(
                f"Semantic retry "
                f"{attempt+1}: {e}"
            )

            time.sleep((1.5 ** attempt) + random.uniform(0.5, 1.5))

    logger.error("Semantic Scholar failed")

    return []