from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.schemas.models import Chunk


def test_vector_store_add_and_search() -> None:
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            source="doc1.txt",
            text="FAISS is used for efficient similarity search over dense vectors.",
            metadata={},
        ),
        Chunk(
            chunk_id="chunk_2",
            source="doc2.txt",
            text="Retrieval augmented generation combines retrieval with text generation.",
            metadata={},
        ),
    ]

    embedder = Embedder()
    records = embedder.embed_chunks(chunks)

    store = VectorStore(embedding_dim=len(records[0].embedding))
    store.add_embeddings(records)

    query_embedding = embedder.embed_texts(["What is FAISS used for?"])[0].tolist()
    results = store.search(query_embedding, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id in {"chunk_1", "chunk_2"}