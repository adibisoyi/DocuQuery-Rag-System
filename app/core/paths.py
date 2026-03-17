from __future__ import annotations

from pathlib import Path

from app.core.config import DEFAULT_INDEX_ROOT


def get_corpus_index_dir(corpus_name: str) -> Path:
    return DEFAULT_INDEX_ROOT / corpus_name


def get_faiss_index_path(corpus_name: str) -> Path:
    return get_corpus_index_dir(corpus_name) / "faiss.index"


def get_records_path(corpus_name: str) -> Path:
    return get_corpus_index_dir(corpus_name) / "records.json"


def get_manifest_path(corpus_name: str) -> Path:
    return get_corpus_index_dir(corpus_name) / "manifest.json"