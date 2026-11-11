# Claude RAG Workspace

This repository contains a learning playground and a FastAPI-based PDF RAG application.

## Main application

The runnable application is in `playground/`.

```bash
cd playground
cp .env.example .env
uv sync
ollama pull nomic-embed-text:latest
ollama pull gemma3:4b
uv run python -m app.ingest
uv run uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to use the API.

See `playground/README.md` for setup, configuration, and troubleshooting.
