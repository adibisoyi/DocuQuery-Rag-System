from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_query_before_indexing_fails() -> None:
    response = client.post("/query", json={"question": "What is FAISS used for?"})
    assert response.status_code == 400
    assert "Call /documents/index first" in response.json()["detail"]


def test_index_documents_endpoint() -> None:
    response = client.post("/documents/index")
    assert response.status_code == 200

    data = response.json()
    assert data["documents_indexed"] >= 1
    assert data["chunks_indexed"] >= 1


def test_query_after_indexing() -> None:
    client.post("/documents/index")
    response = client.post("/query", json={"question": "What is FAISS used for?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) >= 1