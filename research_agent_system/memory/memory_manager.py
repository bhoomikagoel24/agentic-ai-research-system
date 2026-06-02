import hashlib
import json

from research_agent_system.memory.vector_store import (
    VectorStore
)


class MemoryManager:

    def __init__(self):

        self.store = VectorStore()

    def save_plan(
        self,
        topic: str,
        plan: dict
    ):

        text = json.dumps(
            plan,
            ensure_ascii=False
        )

        plan_id = hashlib.md5(
            topic.encode("utf-8")
        ).hexdigest()

        self.store.add_paper(
            paper_id=plan_id,
            text=text,
            metadata={
                "type": "plan",
                "topic": topic
            }
        )
    # ==========================================
    # SAVE SUMMARY
    # ==========================================
    def save_paper(
        self,
        paper: dict
    ):

        text = json.dumps(
            paper,
            ensure_ascii=False
        )

        paper_id = hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()

        self.store.add_paper(
            paper_id=paper_id,
            text=text,
            metadata={
                "type": "paper",
                "title": paper.get(
                    "title",
                    ""
                )
            }
        )
    def save_summary(
        self,
        paper: dict,
        summary: dict
    ):
        # Saving Full JSON summary
        
        paper_text = json.dumps(
            summary,
            ensure_ascii=False
        )

        paper_id = hashlib.md5(
            paper_text.encode("utf-8")
        ).hexdigest()

        metadata = {
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "source": paper.get("source", "")
        }

        self.store.add_paper(
            paper_id=paper_id,
            text=paper_text,
            metadata=metadata
        )

    # ==========================================
    # SEARCH RELATED MEMORY
    # ==========================================

    def search_memory(
        self,
        query: str,
        top_k: int = 3
    ):

        return self.store.search(
            query=query,
            top_k=top_k
        )