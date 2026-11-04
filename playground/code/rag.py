
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore


# ==========================================
# 1. CONFIGURATION
# ==========================================

CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma3:4b"

EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "nomic-embed-text:latest"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

TOP_K = 3
SCORE_THRESHOLD = 0.5


# ==========================================
# 2. LOAD DOCUMENT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "shop_handbook.md"

if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"Document not found: {FILE_PATH}"
    )

text = FILE_PATH.read_text(encoding="utf-8")

if not text.strip():
    raise ValueError("The document is empty!")


doc = Document(
    page_content=text,
    metadata={
        "source": "shop_handbook.md",
        "author": "Cashfeet"
    }
)

print("\nDocument loaded successfully!")


# ==========================================
# 3. TEXT CHUNKING
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents([doc])

print(f"Total chunks created: {len(chunks)}")


# ==========================================
# 4. CREATE EMBEDDINGS
# ==========================================

embeddings = init_embeddings(
    model=EMBEDDING_MODEL,
    provider=EMBEDDING_PROVIDER
)

print("Embeddings model initialized!")


# ==========================================
# 5. VECTOR STORE
# ==========================================

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Vector store created successfully!")


# ==========================================
# 6. INITIALIZE CHAT MODEL
# ==========================================

llm = init_chat_model(
    model=CHAT_MODEL,
    model_provider=CHAT_PROVIDER,
    temperature=0.1
)

print("Chat model initialized!")


# ==========================================
# 7. RAG QUESTION ANSWERING
# ==========================================

def ask_question(query: str):

    if not query.strip():
        print("Please enter a valid question.")
        return

    print("\nSearching relevant documents...\n")

    # Retrieve relevant chunks
    results = vector_store.similarity_search_with_score(
        query,
        k=TOP_K
    )

    # Display similarity scores
    print("Retrieved chunks:")

    for index, (document, score) in enumerate(results, start=1):
        print(f"\nChunk {index}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Content: {document.page_content[:100]}...")

    # Filter relevant chunks
    relevant_texts = []

    for document, score in results:
        if score >= SCORE_THRESHOLD:
            relevant_texts.append(document.page_content)

    # If no relevant chunks found
    if not relevant_texts:
        print("\nAnswer: I don't know.")
        return

    # Combine context
    context = "\n\n".join(relevant_texts)

    # Prepare messages
    messages = [
        (
            "system",
            """
            You are a helpful Cashfeet shop assistant.

            Answer questions using ONLY the provided context.

            Rules:
            1. Do not make up information.
            2. If the answer is not in the context,
               respond with "I don't know".
            3. Give clear and simple answers.
            4. Stay relevant to the user's question.
            """
        ),
        (
            "human",
            f"""
            Context:
            {context}

            Question:
            {query}
            """
        )
    ]

    # Generate answer
    response = llm.invoke(messages)

    print("\n========== ANSWER ==========")
    print(response.content)
    print("============================\n")


# ==========================================
# 8. MAIN PROGRAM
# ==========================================

def main():

    print("\n================================")
    print("       CASHFEET RAG SYSTEM")
    print("================================")

    print("Type 'exit' to quit.")

    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("\nThank you for using Cashfeet RAG!")
            break

        ask_question(query)


if __name__ == "__main__":
    main()
