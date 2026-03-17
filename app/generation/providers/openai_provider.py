from __future__ import annotations

from openai import OpenAI

from app.generation.providers.base import BaseGenerationProvider


class OpenAIProvider(BaseGenerationProvider):
    """
    OpenAI-backed generation provider.
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text.strip()