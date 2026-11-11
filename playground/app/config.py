from functools import lru_cache
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient

PLAYGROUND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PLAYGROUND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_path: str = "code/shop_handbook.pdf"
    chat_model: str = "gemma3:4b"
    embedding_model: str = "nomic-embed-text:latest"
    temperature: float = 0.1
    top_k: int = 3
    chunk_size: int = 600
    chunk_overlap: int = 100
    collection_name: str = "shop_handbook"
    qdrant_path: str = "./qdrant_data"
    system_prompt: str = (
        "You are a helpful shop handbook assistant. Answer using only the "
        "provided context. If the context does not contain the answer, say "
        "that you do not know. Do not invent facts."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def resolve_data_path(settings: Settings) -> Path:
    path = Path(settings.data_path).expanduser()
    if not path.is_absolute():
        path = PLAYGROUND_DIR / path
    return path.resolve()


def get_embeddings(settings: Settings | None = None) -> OllamaEmbeddings:
    settings = settings or get_settings()
    return OllamaEmbeddings(model=settings.embedding_model)


def get_chat_model(settings: Settings | None = None) -> ChatOllama:
    settings = settings or get_settings()
    return ChatOllama(
        model=settings.chat_model,
        temperature=settings.temperature,
    )


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    path = Path(settings.qdrant_path).expanduser()
    if not path.is_absolute():
        path = PLAYGROUND_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path.resolve()))
