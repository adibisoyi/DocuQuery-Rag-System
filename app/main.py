from app.ingestion.loader import DocumentLoader


def main() -> None:
    loader = DocumentLoader(data_dir="data/raw")
    documents = loader.load_documents()

    print(f"Loaded {len(documents)} documents")
    for doc in documents:
        preview = doc.text[:120].replace("\n", " ")
        print(f"- {doc.source}: {preview}...")


if __name__ == "__main__":
    main()