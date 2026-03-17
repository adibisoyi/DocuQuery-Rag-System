from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    question: str


class QueryAPIResponse(BaseModel):
    answer: str
    sources: List[str]


class IndexResponse(BaseModel):
    message: str
    documents_indexed: int
    chunks_indexed: int