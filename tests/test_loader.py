from pathlib import Path

from app.ingestion.loader import DocumentLoader


def test_loader_reads_supported_files_from_temp_dir(tmp_path: Path) -> None:
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("This is a test document.", encoding="utf-8")

    md_file = tmp_path / "notes.md"
    md_file.write_text("# Title\nSome markdown content.", encoding="utf-8")

    loader = DocumentLoader(data_dir=tmp_path)
    documents = loader.load_documents()

    sources = {doc.source for doc in documents}

    assert "sample.txt" in sources
    assert "notes.md" in sources
    assert len(documents) == 2