from app.generation.providers.fallback_provider import FallbackGenerationProvider
from app.generation.providers.ollama_provider import OllamaProvider
from app.generation.providers.openai_provider import OpenAIProvider
from app.generation.providers.provider_factory import get_generation_provider


def test_factory_returns_fallback_when_provider_not_set(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = get_generation_provider()
    assert isinstance(provider, FallbackGenerationProvider)


def test_factory_returns_openai_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")

    provider = get_generation_provider()
    assert isinstance(provider, OpenAIProvider)


def test_factory_returns_ollama_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    provider = get_generation_provider()
    assert isinstance(provider, OllamaProvider)