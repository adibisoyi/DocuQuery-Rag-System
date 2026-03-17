from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="DocuQuery RAG API",
    description="A retrieval-augmented generation backend for querying local documents.",
    version="1.0.0",
)

app.include_router(router)