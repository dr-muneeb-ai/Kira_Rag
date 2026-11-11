# PDF RAG Playground

A local Retrieval-Augmented Generation (RAG) application. It reads a PDF handbook, splits it into chunks, embeds the chunks with Ollama, stores vectors in persistent local Qdrant, retrieves relevant chunks, and asks an Ollama chat model to answer from that context.

## 1. Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) installed and running

## 2. Install

```bash
cd playground
cp .env.example .env
uv sync
ollama pull nomic-embed-text:latest
ollama pull gemma3:4b
```

The sample PDF is at `code/shop_handbook.pdf`. To use another PDF, change `DATA_PATH` in `.env`.

## 3. Index the PDF

```bash
uv run python -m app.ingest
```

This builds/rebuilds the configured Qdrant collection. **Re-indexing with this command replaces the collection named by `COLLECTION_NAME`.** Keep a backup if you need its existing vectors.

## 4. Start the API

```bash
uv run uvicorn app.main:app --reload
```

- API root: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive docs: http://127.0.0.1:8000/docs

Example request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the handbook say about returns?"}'
```

## 5. Important files

```text
playground/
├── .env.example
├── pyproject.toml
├── app/
│   ├── main.py              # FastAPI endpoints
│   ├── config.py            # Settings, Ollama and Qdrant clients
│   ├── ingest.py            # CLI entry point for indexing
│   ├── rag.py               # Retrieval and answer generation
│   └── services/
│       ├── chat.py          # Reusable chat service wrapper
│       └── ingestion.py     # Reusable PDF ingestion service
└── code/
    ├── config.py            # Standalone experiment configuration
    ├── rag.py               # Markdown RAG learning example
    ├── rag-03.py            # PDF RAG learning example
    └── shop_handbook.pdf    # Sample knowledge source
```

## 6. Configuration

All settings can be overridden with environment variables in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `CHAT_MODEL` | `gemma3:4b` | Ollama chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | Ollama embedding model |
| `DATA_PATH` | `code/shop_handbook.pdf` | PDF path, relative to `playground/` |
| `COLLECTION_NAME` | `shop_handbook` | Qdrant collection |
| `QDRANT_PATH` | `./qdrant_data` | Persistent local Qdrant directory |
| `CHUNK_SIZE` | `600` | Maximum chunk size |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `3` | Number of retrieved chunks |
| `TEMPERATURE` | `0.1` | Chat model temperature |

## 7. Troubleshooting

- **Ollama connection error:** start Ollama and check `ollama list`.
- **Model not found:** pull the exact model names configured in `.env`.
- **PDF not found:** verify `DATA_PATH`; relative paths are resolved from the `playground/` directory.
- **Collection missing:** run `uv run python -m app.ingest` before asking questions.
- **No useful answer:** check that the PDF contains selectable text and that the question is covered by it.

## 8. Security

Never commit `.env`, API keys, passwords, local databases, or virtual environments. `.env.example` contains safe defaults only.
