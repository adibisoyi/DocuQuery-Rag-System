from pathlib import Path

from app.retrieval.vector_store import VectorStore
from app.schemas.models import EmbeddingRecord


def test_vector_store_save_and_load(tmp_path: Path) -> None:
    records = [
        EmbeddingRecord(
            chunk_id="chunk_1",
            source="doc1.txt",
            text="FAISS is used for vector similarity search.",
            embedding=[0.1, 0.2, 0.3],
            metadata={},
        )
    ]

    store = VectorStore(embedding_dim=3)
    store.add_embeddings(records)

    index_path = tmp_path / "faiss.index"
    records_path = tmp_path / "records.json"

    store.save(index_path=str(index_path), metadata_path=str(records_path))

    loaded = VectorStore.load(
        embedding_dim=3,
        index_path=str(index_path),
        metadata_path=str(records_path),
    )

    assert loaded.index.ntotal == 1
    assert len(loaded.records) == 1
    assert loaded.records[0].chunk_id == "chunk_1"