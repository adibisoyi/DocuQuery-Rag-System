from pathlib import Path

from app.maintenance.cleanup_indexes import cleanup_indexes
from app.retrieval.index_manifest import save_manifest


def _write_dummy_corpus(index_root: Path, corpus_name: str, last_accessed_at: str) -> None:
    corpus_dir = index_root / corpus_name
    corpus_dir.mkdir(parents=True, exist_ok=True)

    (corpus_dir / "faiss.index").write_bytes(b"12345")
    (corpus_dir / "records.json").write_text("[]", encoding="utf-8")

    save_manifest(
        corpus_dir / "manifest.json",
        {
            "corpus_name": corpus_name,
            "embedding_dim": 384,
            "documents_indexed": 1,
            "chunks_indexed": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_accessed_at": last_accessed_at,
        },
    )


def test_cleanup_removes_ttl_expired_corpus(tmp_path: Path) -> None:
    _write_dummy_corpus(tmp_path, "stale", "2025-01-01T00:00:00+00:00")
    _write_dummy_corpus(tmp_path, "active", "2026-03-17T00:00:00+00:00")

    summary = cleanup_indexes(
        active_corpus_name="active",
        index_root=tmp_path,
        ttl_days=30,
        max_storage_bytes=1024 * 1024,
        enabled=True,
        dry_run=False,
    )

    assert "stale" in summary["deleted_corpora"]
    assert (tmp_path / "active").exists()
    assert not (tmp_path / "stale").exists()


def test_cleanup_respects_dry_run(tmp_path: Path) -> None:
    _write_dummy_corpus(tmp_path, "stale", "2025-01-01T00:00:00+00:00")

    summary = cleanup_indexes(
        active_corpus_name=None,
        index_root=tmp_path,
        ttl_days=30,
        max_storage_bytes=1024 * 1024,
        enabled=True,
        dry_run=True,
    )

    assert "stale" in summary["deleted_corpora"]
    assert (tmp_path / "stale").exists()


def test_cleanup_never_deletes_active_corpus(tmp_path: Path) -> None:
    _write_dummy_corpus(tmp_path, "default", "2025-01-01T00:00:00+00:00")

    summary = cleanup_indexes(
        active_corpus_name="default",
        index_root=tmp_path,
        ttl_days=30,
        max_storage_bytes=1,
        enabled=True,
        dry_run=False,
    )

    assert "default" not in summary["deleted_corpora"]
    assert (tmp_path / "default").exists()