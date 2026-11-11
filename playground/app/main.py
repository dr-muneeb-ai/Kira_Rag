from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import ask_question, get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The API can start before indexing; /ask returns a useful error until
    # the collection is created by `python -m app.ingest`.
    try:
        get_vector_store()
    except RuntimeError as exc:
        app.state.startup_note = str(exc)
    yield


app = FastAPI(
    title="Shop Handbook RAG API",
    description="Ask questions using the indexed shop handbook PDF.",
    version="1.1.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {"message": "Shop Handbook RAG API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        return ask_question(request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid returning internal exception details to API clients.
        raise HTTPException(
            status_code=500,
            detail="The RAG request failed. Check the server logs.",
        ) from exc
