from sentence_transformers import SentenceTransformer, util as st_util
from research_agent_system.utils.logger import get_logger
from research_agent_system.tools.semantic_scholar_tool import fetch_from_semantic_scholar
from research_agent_system.tools.arxiv_tool import fetch_from_arxiv

logger = get_logger(__name__)
_embed_model = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def compute_relevance_score(paper: dict, query: str) -> float:
    text = (paper.get("title","") + " " + paper.get("abstract",""))[:512]
    m = get_embed_model()
    return float(st_util.cos_sim(
        m.encode(text, convert_to_tensor=True),
        m.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    ))


def compute_recency_score(paper: dict) -> float:
    year = paper.get("year")
    if not year or not isinstance(year, int):
        return 0.0
    return 1 / (1 + max(0, 2025 - year))


def filter_papers(papers: list[dict], queries: list[str]) -> list[dict]:
    scored = []
    for p in papers:
        if not p.get("abstract") or len(p["abstract"]) < 100:
            continue
        p["relevance_score"] = max(compute_relevance_score(p, q) for q in queries)
        p["recency_score"]   = compute_recency_score(p)
        scored.append(p)
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:min(10, len(scored))]


def rank_papers(papers: list[dict]) -> list[dict]:
    return sorted(
        papers,
        key=lambda x: 0.6*x.get("relevance_score",0) + 0.4*x.get("recency_score",0),
        reverse=True,
    )


def remove_similar_titles(papers: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for p in papers:
        key = p["title"][:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def multi_source_fetch(query: str) -> list[dict]:
    papers = []
    try:    papers.extend(fetch_from_semantic_scholar(query, 2))
    except Exception as e: logger.warning(f"Semantic skipped: {e}")
    try:    papers.extend(fetch_from_arxiv(query, 2))
    except Exception as e: logger.warning(f"arXiv skipped: {e}")
    return papers