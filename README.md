# DocuQuery - Production-Ready RAG System
![License](https://img.shields.io/github/license/adibisoyi/DocuQuery-Rag-System)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)


A modular, evaluation-driven Retrieval-Augmented Generation (RAG) system built with production-grade design principles including persistence, lifecycle management, and pluggable LLM backends.

## Overview

DocuQuery is not just a demo RAG pipeline - it is designed as a **production-oriented system** with real-world considerations:

- Grounded answer generation  
- Evaluation-driven iteration  
- Persistent vector indexing  
- Lifecycle and cleanup policies  
- API-first modular architecture  

This project reflects how modern ML systems are designed, tested, and deployed in real environments.

---
## Architecture
```text
Documents → Chunking → Embeddings → FAISS Index → Retriever → Generator → Answer
```
### Pipeline Flow
```text
1. Load documents from a corpus  
2. Split documents into token-based chunks  
3. Generate embeddings using SentenceTransformers  
4. Store embeddings in FAISS index  
5. Retrieve relevant chunks via similarity search  
6. Generate grounded responses using an LLM  
```
---
## Key Features

### Grounded Generation

- Answers strictly derived from retrieved context  
- No hallucinations or external assumptions  
- Explicit fallback when context is insufficient  

**Fallback behavior:**
I don’t know based on the provided documents.

---
### Evaluation Framework

Run evaluation: `python -m eval.run_eval`

**Tracks:**

- Source accuracy  
- Answer correctness  
- Retrieval effectiveness  

Used to iteratively improve system performance (e.g., similarity threshold tuning).

---
### Persistent Indexing

Per-corpus index storage:
```text
index/
└── default/
    ├── faiss.index
    ├── records.json
    └── manifest.json
```

**Capabilities:**

- Avoids recomputation of embeddings  
- Loads index on restart  
- Tracks metadata and usage patterns  

**Manifest example:**
```text
{
“corpus_name”: “default”,
“embedding_dim”: 384,
“documents_indexed”: 2,
“chunks_indexed”: 2,
“created_at”: “…”,
“last_accessed_at”: “…”
}
```

---
### Cleanup and Eviction Policy

Environment-based lifecycle configuration:
```text
ENABLE_INDEX_CLEANUP=true
INDEX_TTL_DAYS=30
MAX_INDEX_STORAGE_MB=1024
INDEX_CLEANUP_DRY_RUN=false
```
**Supports:**

- TTL-based deletion of stale corpora  
- Storage-cap eviction (LRU-style)  
- Active corpus protection  
- Dry-run mode for safe validation  

---
### Pluggable LLM Providers

Supports multiple generation backends:
- OpenAI API  
- Ollama (local models)  
- Fallback provider when no LLM is configured  

---
## API Layer

Run server:
`uvicorn app.main:app --reload`
### Endpoints

**Index Documents**
`POST /documents/index`

**Query**
`POST /query`

**Example Request:**
```text
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is FAISS used for?"}'
```
---
## Test Coverage

Run tests:
`pytest`

**Covers:**
- Document ingestion  
- Chunking logic  
- Embeddings pipeline  
- Retrieval system  
- Generator behavior  
- API endpoints  
- Persistence layer  
- Cleanup policies  

---
## Project Structure
```text
app/
├── api/
├── chunking/
├── core/
├── embeddings/
├── generation/
├── ingestion/
├── maintenance/
└── retrieval/
data/
└── corpora/
eval/
└── datasets/
index/
tests/
```

---
## Tech Stack

| Component       | Technology              |
|----------------|------------------------|
| Embeddings     | SentenceTransformers   |
| Vector Store   | FAISS                  |
| Backend        | FastAPI                |
| LLM Providers  | OpenAI / Ollama        |
| Testing        | Pytest                 |
| Evaluation     | Custom Framework       |
| Config         | Environment Variables  |
---

## Getting Started

### Clone Repository
```text
git clone https://github.com/<your-username>/docuquery-rag-system.git
cd docuquery-rag-system
```
### Setup Environment
```text
conda create -n rag python=3.9
conda activate rag
pip install -r requirements.txt
```
### Run API
```text
uvicorn app.main:app --reload
```

### Index Documents
```text
curl -X POST http://127.0.0.1:8000/documents/index
```

### Query
```text
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is FAISS used for?"}'
```
---
## Design Highlights

### Corpus-Aware Architecture
Supports multiple datasets:
```text
data/corpora/<corpus_name>/
index/<corpus_name>/
```
---

### Retrieval Safety
- Similarity thresholding prevents irrelevant matches  
- Improves answer reliability and precision  

---

### Production-Oriented Design
- Persistent indexing  
- Configurable lifecycle policies  
- Modular system design  
- API-first architecture  

---

## Example Output
```text
{
"answer": “FAISS is used for efficient similarity search over dense vector embeddings.”,
"sources": [“notes.md::notes.md_chunk_1”]
}
```
---

## Future Work
- Cross-encoder reranking  
- Hybrid search (BM25 + vector)  
- Document upload API  
- Docker-based deployment  
- Observability (logging, metrics)  

---

## Author
**Aditya Bisoyi**  
Software Engineer focused on scalable systems and applied machine learning.

---
## Contributing
Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
