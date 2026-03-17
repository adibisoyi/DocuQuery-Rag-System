from app.generation.generator import Generator
from app.schemas.models import Chunk, RetrievalResult


def test_generator_returns_grounded_response() -> None:
    results = [
        RetrievalResult(
            chunk=Chunk(
                chunk_id="chunk_1",
                source="doc1.txt",
                text="FAISS is used for efficient similarity search over dense vectors.",
                metadata={},
            ),
            score=0.9,
        )
    ]

    generator = Generator(max_context_chunks=2)
    response = generator.generate("What is FAISS used for?", results)

    assert "FAISS is used" in response.answer
    assert response.sources == ["doc1.txt::chunk_1"]
    assert "QUESTION:" in response.metadata["prompt"]
    assert "CONTEXT:" in response.metadata["prompt"]


def test_generator_returns_fallback_when_no_results() -> None:
    generator = Generator(max_context_chunks=2)
    response = generator.generate("Unknown question", [])

    assert response.answer == "I don't know based on the provided documents."
    assert response.sources == []