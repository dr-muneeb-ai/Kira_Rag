
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings


# ==========================================
# 1. SETTINGS
# ==========================================

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PDF document
    data_path: str = "shop_handbook.pdf"

    # Chat model
    chat_provider: str = "ollama"
    chat_model: str = "gemma4:31b-cloud"

    # Embedding model
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text:latest"
    embedding_dim: int = 768

    # LLM parameters
    temperature: float = 0.1
    top_k: int = 3

    # Chunk settings
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Qdrant settings
    collection_name: str = "shop_handbook"
    qdrant_path: str = "./qdrant_data_notebook"

    # System prompt
    system_prompt: str = (
        "You are a helpful Shop assistant. "
        "Answer only from the provided context. "
        "If the answer is not in the context, say you don't know."
    )


# ==========================================
# 2. GET SETTINGS
# ==========================================

@lru_cache
def get_settings():
    return Settings()


# ==========================================
# 3. QDRANT CLIENT
# ==========================================

def get_qdrant_client(settings: Settings):

    return QdrantClient(
        path=settings.qdrant_path
    )


# ==========================================
# 4. EMBEDDINGS
# ==========================================

def get_embeddings(settings: Settings):

    if settings.embedding_provider == "ollama":

        return OllamaEmbeddings(
            model=settings.embedding_model
        )

    raise ValueError(
        f"Unsupported embedding provider: "
        f"{settings.embedding_provider}"
    )
