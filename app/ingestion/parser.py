from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


def normalize_text(text: str) -> str:
    """
    Normalize extracted text by removing excessive whitespace
    while preserving paragraph boundaries as much as possible.
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def parse_txt(file_path: str | Path) -> str:
    """
    Parse a plain text file.
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return normalize_text(text)


def parse_md(file_path: str | Path) -> str:
    """
    Parse a markdown file as plain text for now.
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return normalize_text(text)


def parse_pdf(file_path: str | Path) -> str:
    """
    Extract text from a PDF using pypdf.
    """
    path = Path(file_path)
    reader = PdfReader(str(path))

    pages_text: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages_text.append(extracted)

    return normalize_text("\n\n".join(pages_text))