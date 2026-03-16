
# DocuQuery: Retrieval-Augmented Knowledge System
### Product Requirements & Technical Design Document

Author: Aditya Bisoyi  
Project Type: Applied ML System  
Status: Design Approved – Implementation Pending

---

# 1. Overview

DocuQuery is a Retrieval-Augmented Generation (RAG) system designed to enable users to query a collection of documents and receive answers grounded in those documents.

Instead of relying solely on a Large Language Model’s internal knowledge, the system retrieves relevant document segments using embeddings and vector similarity search, then uses those retrieved segments to generate answers.

This approach improves factual accuracy, reduces hallucinations, and enables querying over private or proprietary knowledge bases.

The system is designed as a modular Applied ML pipeline consisting of document ingestion, embedding generation, vector indexing, retrieval, and answer generation.

---

# 2. Problem Statement

Large Language Models (LLMs) are powerful but suffer from several limitations:

- hallucinations
- outdated knowledge
- lack of grounding in proprietary documents
- inability to access external datasets without prompting

Organizations often require systems that allow LLMs to answer questions using internal documentation, research papers, or knowledge bases.

Retrieval-Augmented Generation addresses this problem by retrieving relevant document segments before generating answers.

This project builds a production-style RAG system demonstrating the architecture used in modern Applied ML systems.

---

# 3. Goals

## Primary Goals

1. Build an end-to-end Retrieval-Augmented Generation system.
2. Enable querying of technical documents using semantic search.
3. Generate answers grounded in retrieved document context.
4. Provide citations linking answers to document sources.
5. Expose system functionality via a simple API interface.

## Secondary Goals

1. Demonstrate modern Applied ML system design principles.
2. Enable experimentation with retrieval configurations.
3. Provide a modular architecture allowing interchangeable components.
4. Enable future extensions such as hybrid retrieval and reranking.

---

# 4. Non-Goals

The initial version of the system will not include:

- distributed vector databases
- multi-user authentication
- large-scale ingestion pipelines
- hybrid search (BM25 + vector search)
- reranking models
- UI frontend

These capabilities may be added in future iterations.

---

# 5. Target Users

Primary users include:

- machine learning engineers
- software engineers
- researchers
- developers working with technical documentation

Example queries:

- “What is Retrieval-Augmented Generation?”
- “How does FAISS perform similarity search?”
- “What are the advantages of vector databases?”

---

# 6. System Requirements

## Functional Requirements

### FR1 — Document Ingestion
The system must ingest documents from a specified local directory.

Supported formats:

- PDF
- TXT
- Markdown

### FR2 — Document Parsing
Documents must be parsed and converted into clean text.

The system must handle:

- whitespace normalization
- removal of formatting artifacts
- extraction of textual content from PDFs

### FR3 — Chunking Pipeline
Documents must be split into smaller segments called chunks.

Configuration:
chunk_size = 600 tokens  
chunk_overlap = 100 tokens

Each chunk must contain metadata including:

- document source
- chunk identifier
- text content

### FR4 — Embedding Generation
Each chunk must be converted into a vector embedding using a transformer-based embedding model.

The system must support pluggable embedding providers.

### FR5 — Vector Indexing
Embeddings must be stored in a vector index enabling similarity search.

Initial implementation:
FAISS IndexFlatIP

Embeddings must be normalized before indexing.

### FR6 — Query Processing
When a user submits a question:

1. The query is embedded.
2. The vector index retrieves the top-k relevant chunks.
3. Retrieved chunks are passed as context to the generation module.

Default retrieval parameter:
top_k = 5

### FR7 — Answer Generation
The generation module produces an answer using retrieved context.

If the answer is not present in the context, return:
"I don't know based on the provided documents."

### FR8 — Citations
Answers must include references to document chunks used.

Example:

Answer: Retrieval-Augmented Generation improves accuracy by retrieving relevant document context.

Sources:
paper1.pdf – chunk_12  
paper2.pdf – chunk_3

### FR9 — API Interface

POST /documents/index  
POST /query

---

# 7. Architecture Overview

Document Source
    ↓
Document Loader
    ↓
Text Parser
    ↓
Chunking Engine
    ↓
Embedding Engine
    ↓
FAISS Vector Index
    ↓
Query Embedding
    ↓
Retriever
    ↓
Context Builder
    ↓
LLM Generator
    ↓
Answer + Citations

---

# 8. Implementation Phases

## Phase 1 — Project Setup
Objectives:
- initialize repository
- create project structure
- configure dependencies

## Phase 2 — Document Ingestion
Objectives:
- build document loader
- implement PDF parser
- normalize text

## Phase 3 — Chunking Pipeline
Objectives:
- implement chunking engine
- attach metadata

## Phase 4 — Embedding & Indexing
Objectives:
- generate embeddings
- build FAISS index

## Phase 5 — Retrieval Pipeline
Objectives:
- embed user queries
- retrieve top-k chunks

## Phase 6 — Answer Generation
Objectives:
- integrate LLM
- enforce grounded responses
- generate citations

## Phase 7 — API Layer
Objectives:
- build FastAPI endpoints
- connect query pipeline

## Phase 8 — Evaluation
Objectives:
- test system with known questions
- analyze retrieval performance

---

---

# 9. Baseline Module Design (Main Branch Reference)

This section defines the baseline module responsibilities and interfaces that will exist in the `main` branch. These modules will contain minimal working implementations so that future development can occur through feature branches and pull requests.

The purpose of including these designs in the main branch is to provide architectural clarity and enable iterative development without redefining system structure.

## 9.1 Ingestion Layer

Directory:

app/ingestion/

Modules:

loader.py
parser.py

Responsibilities:

- scan the `data/raw` directory
- detect file type
- pass files to the appropriate parser
- return standardized document objects

Example document schema:

```
{
  "source": "paper1.pdf",
  "text": "clean extracted text"
}
```

---

## 9.2 Chunking Layer

Directory:

app/chunking/

Module:

chunker.py

Responsibilities:

- split parsed documents into chunks
- maintain metadata for traceability

Chunk configuration:

```
chunk_size = 600
chunk_overlap = 100
```

Chunk schema:

```
{
  "chunk_id": "paper1_chunk_4",
  "source": "paper1.pdf",
  "text": "chunk text"
}
```

---

## 9.3 Embedding Layer

Directory:

app/embeddings/

Module:

embedder.py

Responsibilities:

- load embedding model
- convert chunks into vector embeddings
- normalize embeddings before indexing

Baseline model:

```
SentenceTransformers
```

Future design supports pluggable providers:

- Local embeddings
- OpenAI embeddings

---

## 9.4 Vector Store

Directory:

app/retrieval/

Modules:

vector_store.py
retriever.py

Responsibilities:

vector_store.py

- create FAISS index
- add embeddings
- persist index
- load index

retriever.py

- embed user query
- perform similarity search
- return top_k chunks

Baseline configuration:

```
top_k = 5
FAISS IndexFlatIP
```

---

## 9.5 Generation Layer

Directory:

app/generation/

Module:

generator.py

Responsibilities:

- build prompt with retrieved chunks
- enforce context grounding
- generate answer
- return citations

Example prompt template:

```
Use only the context provided to answer the question.
If the answer is not in the context, respond with:
"I don't know based on the provided documents."
Cite the chunk sources for each claim.
```

---

## 9.6 API Layer

Directory:

app/api/

Module:

routes.py

Responsibilities:

- expose REST endpoints
- connect API calls to retrieval pipeline

Endpoints:

```
POST /documents/index
POST /query
```

---

## 9.7 Baseline Pipeline Flow

The baseline implementation in the main branch will follow this execution pipeline:

```
Documents → Parser → Chunker → Embedder → FAISS Index

User Query → Query Embedding → Vector Retrieval → Context Assembly → LLM Generation
```

Future branches will iterate on this baseline architecture while maintaining compatibility with this pipeline.

---

# 10. Future Enhancements

Potential improvements:

- hybrid search (BM25 + vector)
- reranking models
- query rewriting
- streaming responses
- UI interface

---

# 10. Knowledge Outcomes

This project demonstrates understanding of:

- Retrieval-Augmented Generation
- vector databases and embeddings
- LLM prompt design
- Applied ML system architecture
- modular ML pipelines
