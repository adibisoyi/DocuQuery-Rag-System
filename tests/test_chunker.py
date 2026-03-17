from app.chunking.chunker import Chunker
from app.schemas.models import Document


def test_chunk_document_returns_chunks() -> None:
    text = "This is a sample document. " * 300
    document = Document(source="sample.txt", text=text)

    chunker = Chunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(document)

    assert len(chunks) > 0
    assert chunks[0].source == "sample.txt"
    assert chunks[0].chunk_id.startswith("sample.txt_chunk_")