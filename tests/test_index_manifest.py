from pathlib import Path

from app.retrieval.index_manifest import (
    load_manifest,
    save_manifest,
    touch_manifest_access_time,
)


def test_manifest_save_load_and_touch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"

    manifest = {
        "corpus_name": "default",
        "embedding_dim": 384,
        "documents_indexed": 2,
        "chunks_indexed": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_accessed_at": "2026-01-01T00:00:00+00:00",
    }

    save_manifest(manifest_path, manifest)
    loaded = load_manifest(manifest_path)

    assert loaded["corpus_name"] == "default"
    assert loaded["embedding_dim"] == 384

    touched = touch_manifest_access_time(manifest_path)
    assert "last_accessed_at" in touched