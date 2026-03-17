from app.chunking.chunker import Chunker
from app.embeddings.embedder import Embedder
from app.ingestion.loader import DocumentLoader


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

    for record in records[:3]:
        print(
            f"- {record.chunk_id} | dim={len(record.embedding)} | source={record.source}"
        )


if __name__ == "__main__":
    main()