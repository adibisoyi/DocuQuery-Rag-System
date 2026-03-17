from app.chunking.chunker import Chunker
from app.ingestion.loader import DocumentLoader


def main() -> None:
    loader = DocumentLoader(data_dir="data/raw")
    documents = loader.load_documents()

    print(f"Loaded {len(documents)} documents")

    chunker = Chunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)

    print(f"Created {len(chunks)} chunks")

    for chunk in chunks[:5]:
        preview = chunk.text[:120].replace("\n", " ")
        print(
            f"- {chunk.chunk_id} | {chunk.metadata['start_token']}:{chunk.metadata['end_token']} | {preview}..."
        )


if __name__ == "__main__":
    main()