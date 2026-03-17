from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from app.schemas.models import EmbeddingRecord, RetrievalResult, Chunk


class VectorStore:
    """
    FAISS-backed vector store for chunk embeddings.
    Uses exact similarity search with IndexFlatIP.
    """

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.records: list[EmbeddingRecord] = []

    def add_embeddings(self, records: List[EmbeddingRecord]) -> None:
        """
        Add embedding records to the FAISS index and metadata store.
        """
        if not records:
            return

        vectors = np.array([record.embedding for record in records], dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings with shape (_, {self.embedding_dim}), "
                f"got {vectors.shape}"
            )

        self.index.add(vectors)
        self.records.extend(records)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """
        Search the FAISS index using a query embedding.
        """
        if self.index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)

        if query.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected query embedding dim {self.embedding_dim}, got {query.shape[1]}"
            )

        scores, indices = self.index.search(query, top_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            record = self.records[idx]
            chunk = Chunk(
                chunk_id=record.chunk_id,
                source=record.source,
                text=record.text,
                metadata=record.metadata,
            )
            results.append(RetrievalResult(chunk=chunk, score=float(score)))

        return results

    def save(self, index_path: str = "index/faiss.index", metadata_path: str = "index/records.json") -> None:
        """
        Persist the FAISS index and metadata records to disk.
        """
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)

        index_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_file))

        serializable_records = [
            {
                "chunk_id": record.chunk_id,
                "source": record.source,
                "text": record.text,
                "embedding": record.embedding,
                "metadata": record.metadata,
            }
            for record in self.records
        ]

        metadata_file.write_text(json.dumps(serializable_records, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        embedding_dim: int,
        index_path: str = "index/faiss.index",
        metadata_path: str = "index/records.json",
    ) -> "VectorStore":
        """
        Load a FAISS index and associated metadata from disk.
        """
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        store = cls(embedding_dim=embedding_dim)
        store.index = faiss.read_index(str(index_file))

        raw_records = json.loads(metadata_file.read_text(encoding="utf-8"))
        store.records = [
            EmbeddingRecord(
                chunk_id=record["chunk_id"],
                source=record["source"],
                text=record["text"],
                embedding=record["embedding"],
                metadata=record.get("metadata", {}),
            )
            for record in raw_records
        ]

        return store