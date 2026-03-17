from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGenerationProvider(ABC):
    """
    Abstract base class for all generation providers.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from a given prompt.
        """
        pass