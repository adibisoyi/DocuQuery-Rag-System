from __future__ import annotations

import os

from dotenv import load_dotenv

from app.generation.providers.base import BaseGenerationProvider
from app.generation.providers.fallback_provider import FallbackGenerationProvider
from app.generation.providers.ollama_provider import OllamaProvider
from app.generation.providers.openai_provider import OpenAIProvider


load_dotenv()


def get_generation_provider() -> BaseGenerationProvider:
    """
    Select generation provider based on environment variables.

    Supported values:
    - LLM_PROVIDER=openai
    - LLM_PROVIDER=ollama

    Falls back to deterministic provider if configuration is missing or invalid.
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

        if api_key:
            return OpenAIProvider(api_key=api_key, model=model)

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3").strip()
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        return OllamaProvider(model=model, base_url=base_url)

    return FallbackGenerationProvider()