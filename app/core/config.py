from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CORPUS_NAME = os.getenv("DOCUQUERY_CORPUS_NAME", "default")
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "corpora" / DEFAULT_CORPUS_NAME

DEFAULT_INDEX_ROOT = PROJECT_ROOT / "index"
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "eval" / "eval_dataset.json"
ENABLE_INDEX_CLEANUP = os.getenv("ENABLE_INDEX_CLEANUP", "true").lower() == "true"
INDEX_TTL_DAYS = int(os.getenv("INDEX_TTL_DAYS", "30"))
MAX_INDEX_STORAGE_MB = int(os.getenv("MAX_INDEX_STORAGE_MB", "1024"))
INDEX_CLEANUP_DRY_RUN = os.getenv("INDEX_CLEANUP_DRY_RUN", "false").lower() == "true"
MAX_INDEX_STORAGE_BYTES = MAX_INDEX_STORAGE_MB * 1024 * 1024

def get_corpus_name_from_path(corpus_path: Path) -> str:
    return corpus_path.name