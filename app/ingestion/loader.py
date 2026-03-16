from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.ingestion.parser import parse_md, parse_pdf, parse_txt
from app.schemas.models import Document


class DocumentLoader:
    """
    Loads supported documents from a directory and converts them
    into standardized Document objects.
    """

    SUPPORTED_EXTENSIONS: dict[str, Callable[[str], str]] = {
        ".txt": parse_txt,
        ".md": parse_md,
        ".pdf": parse_pdf,
    }

    def __init__(self, data_dir: str = "data/raw") -> None:
        self.data_dir = Path(data_dir)

    def load_documents(self) -> list[Document]:
        """
        Scan the data directory, parse supported files, and return documents.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        documents: list[Document] = []

        for file_path in sorted(self.data_dir.iterdir()):
            if not file_path.is_file():
                continue

            parser = self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())
            if parser is None:
                continue

            try:
                text = parser(file_path)
                if not text.strip():
                    continue

                documents.append(
                    Document(
                        source=file_path.name,
                        text=text,
                        metadata={
                            "file_path": str(file_path),
                            "extension": file_path.suffix.lower(),
                        },
                    )
                )
            except Exception as exc:
                # In a production system, we'd log this instead of print.
                print(f"Failed to load {file_path.name}: {exc}")

        return documents