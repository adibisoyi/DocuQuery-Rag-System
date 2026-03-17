from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import IndexResponse, QueryAPIResponse, QueryRequest
from app.chunking.chunker import Chunker
from app.embeddings.embedder import Embedder
from app.generation.generator import Generator
from app.generation.providers.provider_factory import get_generation_provider
from app.ingestion.loader import DocumentLoader
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.core.config import DEFAULT_CORPUS_PATH

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
    loader = DocumentLoader(data_dir=DEFAULT_CORPUS_PATH)
    documents = loader.load_documents()

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found to index.")

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

    retriever = Retriever(embedder=embedder, vector_store=vector_store, top_k=3, min_score_threshold=0.15)
    generator = Generator(provider=get_generation_provider(), max_context_chunks=3)

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