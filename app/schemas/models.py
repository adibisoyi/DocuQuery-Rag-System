from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """
    Represents a parsed source document.
    """
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """
    Represents a chunk derived from a document.
    """
    chunk_id: str
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    Represents one retrieved chunk along with its similarity score.
    """
    chunk: Chunk
    score: float


@dataclass
class QueryResponse:
    """
    Represents the final response returned by the generation layer.
    """
    answer: str
    sources: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)