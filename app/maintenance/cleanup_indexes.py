from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import (
    DEFAULT_INDEX_ROOT,
    ENABLE_INDEX_CLEANUP,
    INDEX_CLEANUP_DRY_RUN,
    INDEX_TTL_DAYS,
    MAX_INDEX_STORAGE_BYTES,
)

from app.retrieval.index_manifest import load_manifest


@dataclass
class CorpusIndexInfo:
    corpus_name: str
    path: Path
    size_bytes: int
    last_accessed_at: datetime | None
    created_at: datetime | None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0

    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _load_corpus_index_info(index_dir: Path) -> CorpusIndexInfo | None:
    if not index_dir.is_dir():
        return None

    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = load_manifest(manifest_path)
    return CorpusIndexInfo(
        corpus_name=index_dir.name,
        path=index_dir,
        size_bytes=_dir_size_bytes(index_dir),
        last_accessed_at=_parse_iso_datetime(manifest.get("last_accessed_at")),
        created_at=_parse_iso_datetime(manifest.get("created_at")),
    )


def _delete_corpus_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    shutil.rmtree(path, ignore_errors=True)


def cleanup_indexes(
    active_corpus_name: str | None = None,
    index_root: Path = DEFAULT_INDEX_ROOT,
    ttl_days: int = INDEX_TTL_DAYS,
    max_storage_bytes: int = MAX_INDEX_STORAGE_BYTES,
    enabled: bool = ENABLE_INDEX_CLEANUP,
    dry_run: bool = INDEX_CLEANUP_DRY_RUN,
) -> Dict[str, Any]:
    """
    Cleanup persisted corpus indexes using:
    1. TTL-based eviction
    2. LRU-style eviction if total storage still exceeds configured limit

    The currently active corpus is never deleted.
    """
    if not enabled:
        return {
            "enabled": False,
            "dry_run": dry_run,
            "deleted_corpora": [],
            "freed_bytes": 0,
            "remaining_bytes": _dir_size_bytes(index_root),
        }

    index_root.mkdir(parents=True, exist_ok=True)

    corpus_infos: List[CorpusIndexInfo] = []
    for child in index_root.iterdir():
        info = _load_corpus_index_info(child)
        if info is not None:
            corpus_infos.append(info)

    deleted_corpora: List[str] = []
    freed_bytes = 0

    now = datetime.now(timezone.utc)
    ttl_cutoff = now - timedelta(days=ttl_days)

    remaining: List[CorpusIndexInfo] = []

    # Step 1: TTL eviction
    for info in corpus_infos:
        if info.corpus_name == active_corpus_name:
            remaining.append(info)
            continue

        if info.last_accessed_at is not None and info.last_accessed_at < ttl_cutoff:
            deleted_corpora.append(info.corpus_name)
            freed_bytes += info.size_bytes
            _delete_corpus_dir(info.path, dry_run=dry_run)
        else:
            remaining.append(info)

    # Step 2: LRU-style eviction if storage cap exceeded
    current_total = sum(info.size_bytes for info in remaining)

    # oldest first, None treated as very old
    remaining.sort(
        key=lambda x: x.last_accessed_at or datetime.min.replace(tzinfo=timezone.utc)
    )

    kept: List[CorpusIndexInfo] = []
    for info in remaining:
        if info.corpus_name == active_corpus_name:
            kept.append(info)
            continue

        if current_total <= max_storage_bytes:
            kept.append(info)
            continue

        deleted_corpora.append(info.corpus_name)
        freed_bytes += info.size_bytes
        current_total -= info.size_bytes
        _delete_corpus_dir(info.path, dry_run=dry_run)

    # active corpus and undeleted corpora remain
    remaining_bytes = _dir_size_bytes(index_root) if not dry_run else max(
        0, _dir_size_bytes(index_root) - freed_bytes
    )

    return {
        "enabled": True,
        "dry_run": dry_run,
        "deleted_corpora": deleted_corpora,
        "freed_bytes": freed_bytes,
        "remaining_bytes": remaining_bytes,
    }