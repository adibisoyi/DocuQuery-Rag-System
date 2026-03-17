from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.chunking.chunker import Chunker
from app.embeddings.embedder import Embedder
from app.generation.generator import Generator
from app.generation.providers.provider_factory import get_generation_provider
from app.ingestion.loader import DocumentLoader
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


def build_pipeline() -> tuple[Retriever, Generator]:
    loader = DocumentLoader(data_dir="data/raw")
    documents = loader.load_documents()

    chunker = Chunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)

    embedder = Embedder()
    records = embedder.embed_chunks(chunks)

    if not records:
        raise RuntimeError("No embedding records generated during evaluation setup.")

    vector_store = VectorStore(embedding_dim=len(records[0].embedding))
    vector_store.add_embeddings(records)

    retriever = Retriever(embedder=embedder, vector_store=vector_store, top_k=3)
    generator = Generator(provider=get_generation_provider(), max_context_chunks=3)

    return retriever, generator


def load_eval_dataset(path: str = "eval/eval_dataset.json") -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def evaluate_case(case: Dict[str, Any], retriever: Retriever, generator: Generator) -> Dict[str, Any]:
    question = case["question"]
    expected_source = case["expected_source"]
    expected_answer_contains = case["expected_answer_contains"]

    results = retriever.retrieve(question)
    response = generator.generate(question, results)

    actual_source = response.sources[0] if response.sources else None
    source_match = actual_source == expected_source

    answer_text = response.answer.lower()
    answer_match = all(expected.lower() in answer_text for expected in expected_answer_contains)

    return {
        "question": question,
        "expected_source": expected_source,
        "actual_source": actual_source,
        "source_match": source_match,
        "answer": response.answer,
        "answer_match": answer_match,
    }


def main() -> None:
    retriever, generator = build_pipeline()
    dataset = load_eval_dataset()

    results = [evaluate_case(case, retriever, generator) for case in dataset]

    total = len(results)
    source_correct = sum(1 for r in results if r["source_match"])
    answer_correct = sum(1 for r in results if r["answer_match"])

    print("\n=== Evaluation Summary ===")
    print(f"Total cases: {total}")
    print(f"Source accuracy: {source_correct}/{total} = {source_correct / total:.2%}")
    print(f"Answer match:   {answer_correct}/{total} = {answer_correct / total:.2%}")

    print("\n=== Detailed Results ===")
    for idx, result in enumerate(results, start=1):
        print(f"\nCase {idx}")
        print(f"Question:       {result['question']}")
        print(f"Expected source:{result['expected_source']}")
        print(f"Actual source:  {result['actual_source']}")
        print(f"Source match:   {result['source_match']}")
        print(f"Answer match:   {result['answer_match']}")
        print(f"Answer:         {result['answer']}")


if __name__ == "__main__":
    main()