from __future__ import annotations

from typing import List

from app.generation.providers.base import BaseGenerationProvider
from app.generation.providers.fallback_provider import FallbackGenerationProvider
from app.schemas.models import QueryResponse, RetrievalResult


class Generator:
    """
    Handles prompt construction and delegates generation to a provider.
    """

    def __init__(
        self,
        provider: BaseGenerationProvider | None = None,
        max_context_chunks: int = 3,
    ) -> None:
        self.provider = provider or FallbackGenerationProvider()
        self.max_context_chunks = max_context_chunks

    def build_context(self, results: List[RetrievalResult]) -> str:
        selected = results[: self.max_context_chunks]

        context_blocks = []
        for result in selected:
            context_blocks.append(
                f"[SOURCE: {result.chunk.source} | CHUNK: {result.chunk.chunk_id}]\n"
                f"{result.chunk.text}"
            )

        return "\n\n".join(context_blocks)

    def build_prompt(self, question: str, results: List[RetrievalResult]) -> str:
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
        if not results:
            return QueryResponse(
                answer="I don't know based on the provided documents.",
                sources=[],
                metadata={"prompt": self.build_prompt(question, results)},
            )

        prompt = self.build_prompt(question, results)

        # Call provider
        provider_output = self.provider.generate(prompt)

        # Still use top chunk as fallback grounding
        top_result = results[0]
        grounded_answer = top_result.chunk.text.strip()

        sources = [
            f"{result.chunk.source}::{result.chunk.chunk_id}"
            for result in results[: self.max_context_chunks]
        ]

        return QueryResponse(
            answer=f"{provider_output}\n\n{grounded_answer}",
            sources=sources,
            metadata={"prompt": prompt},
        )