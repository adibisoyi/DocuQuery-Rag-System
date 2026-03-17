from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CORPUS_NAME = os.getenv("DOCUQUERY_CORPUS_NAME", "default")
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "corpora" / DEFAULT_CORPUS_NAME

DEFAULT_INDEX_ROOT = PROJECT_ROOT / "index"
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "eval" / "eval_dataset.json"