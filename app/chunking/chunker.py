from __future__ import annotations

from typing import List

from transformers import AutoTokenizer

from app.schemas.models import Chunk, Document


class Chunker:
    """
    Token-aware chunker that splits documents into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk a list of documents into overlapping token-based chunks.
        """
        chunks: List[Chunk] = []

        for document in documents:
            doc_chunks = self.chunk_document(document)
            chunks.extend(doc_chunks)

        return chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk a single document into overlapping token-based chunks.
        """
        token_ids = self.tokenizer.encode(document.text, add_special_tokens=False)

        if not token_ids:
            return []

        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap

        for start_idx in range(0, len(token_ids), step):
            end_idx = start_idx + self.chunk_size
            chunk_token_ids = token_ids[start_idx:end_idx]

            if not chunk_token_ids:
                continue

            chunk_text = self.tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if not chunk_text:
                continue

            chunk_number = len(chunks) + 1
            chunk_id = f"{document.source}_chunk_{chunk_number}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=document.source,
                    text=chunk_text,
                    metadata={
                        "start_token": start_idx,
                        "end_token": min(end_idx, len(token_ids)),
                        "chunk_size": len(chunk_token_ids),
                        "document_metadata": document.metadata,
                    },
                )
            )

            if end_idx >= len(token_ids):
                break

        return chunks