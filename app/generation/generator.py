from __future__ import annotations

from typing import List

from app.schemas.models import QueryResponse, RetrievalResult


class Generator:
    """
    Builds a grounded response from retrieved chunks.

    This baseline implementation does not yet call an external LLM.
    It formats retrieved evidence into a deterministic answer so the
    pipeline remains testable before API integration.
    """

    def __init__(self, max_context_chunks: int = 3) -> None:
        self.max_context_chunks = max_context_chunks

    def build_context(self, results: List[RetrievalResult]) -> str:
        """
        Build a text context block from top retrieval results.
        """
        selected = results[: self.max_context_chunks]

        context_blocks = []
        for result in selected:
            context_blocks.append(
                f"[SOURCE: {result.chunk.source} | CHUNK: {result.chunk.chunk_id}]\n"
                f"{result.chunk.text}"
            )

        return "\n\n".join(context_blocks)

    def build_prompt(self, question: str, results: List[RetrievalResult]) -> str:
        """
        Construct the grounded prompt that would be sent to an LLM.
        """
        context = self.build_context(results)

        return (
            "You are a document-grounded assistant.\n"
            "Use only the provided context to answer the question.\n"
            'If the answer is not in the context, say: "I don\'t know based on the provided documents."\n'
            "Cite the source chunk IDs for important claims.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}"
        )

    def generate(self, question: str, results: List[RetrievalResult]) -> QueryResponse:
        """
        Baseline generation method.

        For now, returns a deterministic grounded response using retrieved chunks.
        This keeps the architecture correct and testable before integrating an LLM provider.
        """
        if not results:
            return QueryResponse(
                answer="I don't know based on the provided documents.",
                sources=[],
                metadata={"prompt": self.build_prompt(question, results)},
            )

        top_result = results[0]
        answer = top_result.chunk.text.strip()

        sources = [f"{result.chunk.source}::{result.chunk.chunk_id}" for result in results[: self.max_context_chunks]]

        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata={"prompt": self.build_prompt(question, results)},
        )