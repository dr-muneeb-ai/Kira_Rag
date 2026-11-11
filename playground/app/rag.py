from functools import lru_cache

from langchain_qdrant import QdrantVectorStore

from app.config import (
    get_chat_model,
    get_embeddings,
    get_qdrant_client,
    get_settings,
)


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    client = get_qdrant_client(settings)

    if not client.collection_exists(settings.collection_name):
        client.close()
        raise RuntimeError(
            "Qdrant collection does not exist. Run "
            "`uv run python -m app.ingest` from the playground directory first."
        )

    # Keep the client alive with the cached vector store for the API lifetime.
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(settings),
    )


def ask_question(question: str) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    settings = get_settings()
    vector_store = get_vector_store()
    documents = vector_store.similarity_search(question, k=settings.top_k)

    if not documents:
        return {
            "question": question,
            "answer": "I don't know based on the available handbook.",
            "sources": [],
        }

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"Page: {doc.metadata.get('page', 'unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in documents
    )

    prompt = (
        f"{settings.system_prompt}\n\n"
        f"Use the context below to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = get_chat_model(settings).invoke(prompt)

    sources = [
        {
            "source": str(doc.metadata.get("source", "unknown")),
            "page": (
                doc.metadata["page"] + 1
                if isinstance(doc.metadata.get("page"), int)
                else None
            ),
            "excerpt": doc.page_content[:300],
        }
        for doc in documents
    ]

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
    }
