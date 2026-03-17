from __future__ import annotations

from typing import List

from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.schemas.models import RetrievalResult


class Retriever:
    """
    Handles query embedding and vector search over the FAISS index.
    Applies a minimum similarity threshold to reject low-confidence matches.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        min_score_threshold: float = 0.15,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score_threshold = min_score_threshold

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Embed the user query and return the top-k retrieval results.
        Reject all results if the best score is below the configured threshold.
        """
        if not query or not query.strip():
            return []

        query_embedding = self.embedder.embed_texts([query])[0].tolist()
        results = self.vector_store.search(query_embedding=query_embedding, top_k=self.top_k)

        if not results:
            return []

        top_score = results[0].score
        if top_score < self.min_score_threshold:
            return []

        return results