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
            "You must answer ONLY using the provided context.\n"
            "Do NOT use prior knowledge.\n"
            "Do NOT make assumptions.\n"
            "Do NOT expand abbreviations or acronyms unless the expansion is explicitly present in the context.\n"
            "Do NOT add definitions, examples, or extra uses unless they are explicitly supported by the context.\n"
            'If the answer is not explicitly supported by the context, say: "I don\'t know based on the provided documents."\n'
            "Return a concise answer and cite the supporting chunk IDs for important claims.\n\n"
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

        # Return only the primary source used for answer generation
        primary_result = results[0]
        sources = [f"{primary_result.chunk.source}::{primary_result.chunk.chunk_id}"]

        return QueryResponse(
            answer=provider_output,
            sources=sources,
            metadata={"prompt": prompt},
        )