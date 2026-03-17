from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.models import Chunk, EmbeddingRecord


class Embedder:
    """
    Embeds chunk text using a SentenceTransformer model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate normalized embeddings for a list of texts.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return self._normalize(embeddings)

    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddingRecord]:
        """
        Convert chunks into EmbeddingRecord objects.
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        records: List[EmbeddingRecord] = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append(
                EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    embedding=embedding.tolist(),
                    metadata=chunk.metadata,
                )
            )

        return records

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings row-wise.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms