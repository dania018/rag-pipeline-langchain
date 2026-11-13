# RAG Pipeline with LangChain

A **Retrieval-Augmented Generation (RAG)** pipeline built from scratch that answers questions using your own PDF documents. It runs LangChain, Sentence Transformers, ChromaDB and a Groq-hosted LLM.

## Why RAG?

Large Language Models have two well-known limits:

1. **Hallucination:** they can make things up.
2. **Stale knowledge:** they only know their training data.

RAG fixes both by **retrieving relevant information from your documents first**, then giving it to the LLM as context. The answers are more accurate, more current, and grounded in facts.

## How it works

The pipeline has two phases.

### 1. Ingestion pipeline (run once)

```
Data → Documents → Chunks → Embeddings → Vector DB
```

| Step | What happens | Tool |
|---|---|---|
| **Load** | Every PDF in `data/pdfs/` is loaded into LangChain `Document` objects (`page_content` + `metadata`) | `PyPDFLoader` |
| **Chunk** | Documents are split into 500-character chunks with a 50-character overlap, so context isn't cut mid-sentence | `RecursiveCharacterTextSplitter` |
| **Embed** | Each chunk becomes a 384-dimensional vector | `all-MiniLM-L6-v2` (Sentence Transformers) |
| **Store** | Vectors, text and metadata are saved in a persistent vector database, the knowledge base | ChromaDB |

### 2. Retrieval + generation pipeline (every question)

```
Query → Query embedding → Retrieve top-k → Augment prompt → LLM → Answer
```

| Step | What happens |
|---|---|
| **Query processing** | The question is embedded with the **same** model used for the chunks |
| **Retrieval** | Semantic search returns the top-k most similar chunks (similarity = 1 − cosine distance) |
| **Augmentation** | The retrieved chunks are added to the prompt as context |
| **Generation** | The LLM (`openai/gpt-oss-120b` via Groq) writes an answer grounded in that context |

## Code structure

The notebook is organised into small, reusable classes:

| Component | Role |
|---|---|
| `load_all_pdfs()` | Loads every PDF in a folder |
| `split_docs()` | Splits documents into overlapping chunks |
| `EmbeddingManager` | Loads the embedding model and turns text into vectors |
| `VectorStoreManager` | Creates the persistent ChromaDB collection and stores chunks + embeddings |
| `RAGRetriever` | Embeds a query and returns the top-k similar chunks with scores |
| `generate_output()` | Combines retrieved context + query into a prompt and calls the LLM |

## Project structure

```
rag-pipeline-langchain/
├── RAG_pipeline.ipynb     # Full pipeline: ingestion, retrieval, generation
├── data/
│   ├── pdfs/              # Put your PDF files here
│   └── Python.txt         # Sample text file for the TextLoader demo
└── README.md
```

`data/vector_store/` is **not** included. It's created automatically the first time you run the notebook.

## Tech stack

Python · LangChain · Sentence Transformers · ChromaDB · Groq · PyPDF
