from __future__ import annotations

from fastapi import APIRouter, HTTPException

from pathlib import Path
from app.maintenance.cleanup_indexes import cleanup_indexes

from app.core.config import DEFAULT_CORPUS_PATH, get_corpus_name_from_path
from app.core.paths import get_faiss_index_path, get_manifest_path, get_records_path
from app.retrieval.index_manifest import (
    load_manifest,
    save_manifest,
    touch_manifest_access_time,
    utc_now_iso,
)
from app.api.schemas import IndexResponse, QueryAPIResponse, QueryRequest
from app.chunking.chunker import Chunker
from app.embeddings.embedder import Embedder
from app.generation.generator import Generator
from app.generation.providers.provider_factory import get_generation_provider
from app.ingestion.loader import DocumentLoader
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


router = APIRouter()

APP_STATE: dict = {
    "is_indexed": False,
    "documents": [],
    "chunks": [],
    "embedder": None,
    "vector_store": None,
    "retriever": None,
    "generator": None,
}


@router.post("/documents/index", response_model=IndexResponse)
def index_documents() -> IndexResponse:
    corpus_path = Path(DEFAULT_CORPUS_PATH)
    corpus_name = get_corpus_name_from_path(corpus_path)
    cleanup_summary = cleanup_indexes(active_corpus_name=corpus_name)

    faiss_index_path = get_faiss_index_path(corpus_name)
    records_path = get_records_path(corpus_name)
    manifest_path = get_manifest_path(corpus_name)

    loader = DocumentLoader(data_dir=corpus_path)
    documents = loader.load_documents()

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found to index.")

    generator = Generator(provider=get_generation_provider(), max_context_chunks=3)

    # Load persisted index if it already exists
    if faiss_index_path.exists() and records_path.exists() and manifest_path.exists():
        manifest = touch_manifest_access_time(manifest_path)
        embedding_dim = manifest.get("embedding_dim")

        if embedding_dim is None:
            raise HTTPException(
                status_code=500,
                detail="Persisted index manifest is missing embedding_dim.",
            )

        embedder = Embedder()
        vector_store = VectorStore.load(
            embedding_dim=embedding_dim,
            index_path=str(faiss_index_path),
            metadata_path=str(records_path),
        )

        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=3,
            min_score_threshold=0.15,
        )

        APP_STATE["is_indexed"] = True
        APP_STATE["documents"] = documents
        APP_STATE["chunks"] = []
        APP_STATE["embedder"] = embedder
        APP_STATE["vector_store"] = vector_store
        APP_STATE["retriever"] = retriever
        APP_STATE["generator"] = generator

        return IndexResponse(
            message="Persisted index loaded successfully.",
            documents_indexed=len(documents),
            chunks_indexed=len(vector_store.records),
        )

    # Build a fresh index
    chunker = Chunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from documents.")

    embedder = Embedder()
    records = embedder.embed_chunks(chunks)

    if not records:
        raise HTTPException(status_code=400, detail="No embeddings generated.")

    vector_store = VectorStore(embedding_dim=len(records[0].embedding))
    vector_store.add_embeddings(records)

    # Overwrite-in-place for this corpus
    vector_store.save(
        index_path=str(faiss_index_path),
        metadata_path=str(records_path),
    )

    now = utc_now_iso()
    save_manifest(
        manifest_path,
        {
            "corpus_name": corpus_name,
            "embedding_dim": len(records[0].embedding),
            "documents_indexed": len(documents),
            "chunks_indexed": len(chunks),
            "created_at": now,
            "last_accessed_at": now,
        },
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=3,
        min_score_threshold=0.15,
    )

    APP_STATE["is_indexed"] = True
    APP_STATE["documents"] = documents
    APP_STATE["chunks"] = chunks
    APP_STATE["embedder"] = embedder
    APP_STATE["vector_store"] = vector_store
    APP_STATE["retriever"] = retriever
    APP_STATE["generator"] = generator

    return IndexResponse(
        message="Documents indexed successfully.",
        documents_indexed=len(documents),
        chunks_indexed=len(chunks),
    )


@router.post("/query", response_model=QueryAPIResponse)
def query_documents(request: QueryRequest) -> QueryAPIResponse:
    if not APP_STATE["is_indexed"]:
        raise HTTPException(
            status_code=400,
            detail="Documents are not indexed yet. Call /documents/index first.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    retriever: Retriever = APP_STATE["retriever"]
    generator: Generator = APP_STATE["generator"]

    results = retriever.retrieve(question)
    response = generator.generate(question, results)

    return QueryAPIResponse(
        answer=response.answer,
        sources=response.sources,
    )