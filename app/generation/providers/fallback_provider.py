from __future__ import annotations

from app.generation.providers.base import BaseGenerationProvider


class FallbackGenerationProvider(BaseGenerationProvider):
    """
    Deterministic fallback provider used when no LLM is available.
    """

    def generate(self, prompt: str) -> str:
        """
        Returns a placeholder response indicating fallback mode.
        """
        return (
            "Fallback response: Unable to use an LLM provider. "
            "Returning the most relevant retrieved content."
        )