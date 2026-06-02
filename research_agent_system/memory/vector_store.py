import chromadb

from sentence_transformers import (
    SentenceTransformer
)

from research_agent_system.utils.logger import (
    get_logger
)

logger = get_logger(__name__)

_embed_model = None

def get_embed_model():

    global _embed_model

    if _embed_model is None:

        _embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return _embed_model

class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="memory/chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="research_memory"
        )

        logger.info(
            "VectorStore initialized"
        )

    # ==========================================
    # EMBEDDING
    # ==========================================

    def embed(self, text: str):

        model = get_embed_model()

        return model.encode(
            text
        ).tolist()
    # ==========================================
    # STORE PAPER
    # ==========================================

    def add_paper(
        self,
        paper_id: str,
        text: str,
        metadata: dict
    ):

        embedding = self.embed(text)

        try:

            self.collection.upsert(
                ids=[paper_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )

        except Exception:

            logger.info(
                "Memory already exists"
            )

        logger.info(
            f"Stored paper: {paper_id}"
        )

    # ==========================================
    # SEARCH MEMORY
    # ==========================================

    def search(
        self,
        query: str,
        top_k: int = 3
    ):

        query_embedding = self.embed(query)

        count = self.collection.count()

        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        output = []

        docs = results.get(
            "documents",
            [[]]
        )[0]

        metas = results.get(
            "metadatas",
            [[]]
        )[0]

        dists = results.get(
            "distances",
            [[]]
        )[0]

        for doc, meta, dist in zip(
            docs,
            metas,
            dists
        ):

            similarity = max(
                0,
                1 - dist
            )

            output.append({
                "document": doc,
                "similarity": similarity
            })

        return output