# Production RAG System

**LangGraph · Pinecone · SentenceTransformers · Groq**

---

## Architecture

```
START
  │
  ▼
router ──── ingest_flag=True ──► pdf_loader ► chunking ► embedding ► vector_store ► END
  │
  └── ingest_flag=False ──────► query_embedding ► retrieval ► answer_generation ► END
```

### Key Design Decisions

| Decision | Detail |
|---|---|
| **No re-embedding on query** | Query embedding runs every time; document embedding only during ingestion |
| **Namespace-per-sector** | Pinecone namespaces map 1:1 to sectors (e.g., `ESDM`, `cement`) |
| **Multi-namespace retrieval** | Results from multiple namespaces are merged and re-ranked globally |
| **Fallback on empty results** | If score threshold yields 0 results, threshold is lowered to 0.0 and retried |
| **YAML prompt** | Structured prompt minimises hallucination by grounding the LLM in context |
| **Groq retry** | Exponential back-off on rate-limit and 5xx errors (configurable max_retries) |
| **LangSmith on every node** | `@traced_node` decorator wraps all nodes with timing + metadata logging |

---

## Project Structure

```
rag_system/
├── api.py                        # FastAPI application
├── main.py                       # Entrypoint (server + CLI)
├── requirements.txt
├── .env.example
├── config/
│   ├── __init__.py
│   └── settings.py               # Pydantic Settings (env vars)
├── graph/
│   ├── __init__.py
│   ├── state.py                  # RAGState dataclass
│   ├── nodes.py                  # All LangGraph nodes
│   └── rag_graph.py              # Conditional graph builder
├── observability/
│   ├── __init__.py
│   └── langsmith.py              # @traced_node decorator + logging
└── tools/
    ├── __init__.py
    ├── st_model.py               # SentenceTransformer singleton
    ├── pdf_loader_tool.py        # Stage 1 — PDF → pages
    ├── chunking_tool.py          # Stage 2 — pages → chunks
    ├── embedding_tool.py         # query & document embedding
    ├── vector_store_tool.py      # Pinecone upsert + query
    ├── retrieval_tool.py         # Pinecone query wrapper
    ├── answer_generation_tool.py # YAML prompt + Groq LLM
    └── tool_analysis.py          # Pipeline output analytics
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Fill in GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
```

### 3. Run the API server

```bash
python main.py serve
# or
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 4. Query (retrieval mode)

```bash
# Via CLI
python main.py query "What is the growth forecast for the ESDM sector?" --namespaces ESDM

# Via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the ESDM market size?", "namespaces": ["ESDM"], "top_k": 5}'
```

### 5. Ingest PDFs (ingestion mode — only when needed)

```bash
# Via CLI
python main.py ingest report1.pdf report2.pdf --sector ESDM

# Via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_paths": ["/data/esdm_report.pdf"], "sector": "ESDM"}'
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Groq API key |
| `PINECONE_API_KEY` | **required** | Pinecone API key |
| `PINECONE_INDEX_NAME` | **required** | Pinecone index name |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `EMBEDDING_DIMENSION` | `384` | Must match index dimension |
| `RETRIEVAL_TOP_K` | `5` | Chunks per query |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.3` | Min cosine similarity |
| `DEFAULT_NAMESPACES` | `["ESDM","cement"]` | Namespaces searched by default |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `LANGCHAIN_API_KEY` | _(optional)_ | Enables LangSmith tracing |
| `LANGCHAIN_PROJECT` | `rag-system` | LangSmith project name |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/namespaces` | List default namespaces |
| `POST` | `/query` | Retrieval-mode question answering |
| `POST` | `/ingest` | PDF ingestion into Pinecone |

### POST /query

```json
{
  "query": "What is the market size of the cement sector?",
  "namespaces": ["cement"],
  "top_k": 5
}
```

### POST /ingest

```json
{
  "pdf_paths": ["/data/cement_report_2024.pdf"],
  "sector": "cement"
}
```

---

## LangSmith Tracing

Set `LANGCHAIN_API_KEY` to enable full tracing. Each node emits:

- **Run name**: `pdf_loader_node`, `chunking_node`, `embedding_node`, `vector_store_node`, `query_embedding_node`, `retrieval_node`, `answer_generation_node`
- **Metadata**: query, namespace(s), retrieved_docs_count, elapsed time
- **Pipeline tag**: `ingestion` or `retrieval`