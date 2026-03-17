from app.embeddings.embedder import Embedder
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas.models import Chunk


def test_retriever_returns_ranked_results() -> None:
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
            text="Retrieval augmented generation combines retrieval and language generation.",
            metadata={},
        ),
    ]

    embedder = Embedder()
    records = embedder.embed_chunks(chunks)

    store = VectorStore(embedding_dim=len(records[0].embedding))
    store.add_embeddings(records)

    retriever = Retriever(embedder=embedder, vector_store=store, top_k=2)
    results = retriever.retrieve("What is FAISS used for?")

    assert len(results) > 0
    assert results[0].chunk.chunk_id == "chunk_1"


def test_retriever_returns_empty_for_blank_query() -> None:
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            source="doc1.txt",
            text="FAISS is used for efficient similarity search over dense vectors.",
            metadata={},
        )
    ]

    embedder = Embedder()
    records = embedder.embed_chunks(chunks)

    store = VectorStore(embedding_dim=len(records[0].embedding))
    store.add_embeddings(records)

    retriever = Retriever(embedder=embedder, vector_store=store, top_k=2)
    results = retriever.retrieve("   ")

    assert results == []