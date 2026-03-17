from app.embeddings.embedder import Embedder
from app.schemas.models import Chunk


def test_embed_chunks_returns_records() -> None:
    chunks = [
        Chunk(
            chunk_id="sample_chunk_1",
            source="sample.txt",
            text="Retrieval augmented generation improves answer quality.",
            metadata={},
        )
    ]

    embedder = Embedder()
    records = embedder.embed_chunks(chunks)

    assert len(records) == 1
    assert records[0].chunk_id == "sample_chunk_1"
    assert records[0].source == "sample.txt"
    assert len(records[0].embedding) > 0