from __future__ import annotations

import requests

from app.generation.providers.base import BaseGenerationProvider


class OllamaProvider(BaseGenerationProvider):
    """
    Local Ollama-backed generation provider.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"

        response = requests.post(
            url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()