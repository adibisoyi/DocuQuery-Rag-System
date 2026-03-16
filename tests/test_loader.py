from app.ingestion.loader import DocumentLoader


def test_loader_initialization() -> None:
    loader = DocumentLoader(data_dir="data/raw")
    assert str(loader.data_dir).endswith("data/raw")