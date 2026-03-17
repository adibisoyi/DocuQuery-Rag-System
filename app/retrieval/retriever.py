from __future__ import annotations

from typing import List

from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.schemas.models import RetrievalResult


class Retriever:
    """
    Handles query embedding and vector search over the FAISS index.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = 5) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Embed the user query and return the top-k retrieval results.
        """
        if not query or not query.strip():
            return []

        query_embedding = self.embedder.embed_texts([query])[0].tolist()
        return self.vector_store.search(query_embedding=query_embedding, top_k=self.top_k)