from app.chunking.chunker import Chunker
from app.embeddings.embedder import Embedder
from app.generation.generator import Generator
from app.generation.providers.provider_factory import get_generation_provider
from app.ingestion.loader import DocumentLoader
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


def main() -> None:
    loader = DocumentLoader(data_dir="data/raw")
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents")

    chunker = Chunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embedder = Embedder()
    records = embedder.embed_chunks(chunks)
    print(f"Generated {len(records)} embedding records")

    if not records:
        print("No embeddings generated. Exiting.")
        return

    vector_store = VectorStore(embedding_dim=len(records[0].embedding))
    vector_store.add_embeddings(records)
    print(f"Indexed {len(vector_store.records)} records in FAISS")

    retriever = Retriever(embedder=embedder, vector_store=vector_store, top_k=3)
    provider = get_generation_provider()
    generator = Generator(provider=provider, max_context_chunks=3)

    query = "What is FAISS used for?"
    results = retriever.retrieve(query)
    response = generator.generate(query, results)

    print(f"\nQuery: {query}")
    print("\nAnswer:")
    print(response.answer)

    print("\nSources:")
    for source in response.sources:
        print(f"- {source}")


if __name__ == "__main__":
    main()