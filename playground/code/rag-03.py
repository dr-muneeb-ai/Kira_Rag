
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore


# =====================================
# 1. CONFIGURATION
# =====================================

CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma3:4b"

EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "nomic-embed-text:latest"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

TOP_K = 3
SCORE_THRESHOLD = 0.5


# =====================================
# 2. PDF LOADER
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "shop_handbook.pdf"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"PDF file not found: {DATA_PATH}"
    )

loader = PyPDFLoader(str(DATA_PATH))

pages = loader.load_and_split()

if not pages:
    raise ValueError("PDF document is empty!")

print(f"\nPDF loaded successfully!")
print(f"Total pages: {len(pages)}")


# =====================================
# 3. TEXT SPLITTER
# =====================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(pages)

if not chunks:
    raise ValueError("No chunks generated!")

print(f"Total chunks: {len(chunks)}")


# =====================================
# 4. EMBEDDINGS
# =====================================

embeddings = init_embeddings(
    model=EMBEDDING_MODEL,
    provider=EMBEDDING_PROVIDER
)

print("\nEmbeddings initialized!")


# =====================================
# 5. VECTOR STORE
# =====================================

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Vector store created!")


# =====================================
# 6. CHAT MODEL
# =====================================

llm = init_chat_model(
    model=CHAT_MODEL,
    model_provider=CHAT_PROVIDER,
    temperature=0.1
)

print("LLM initialized!")


# =====================================
# 7. RAG FUNCTION
# =====================================

def ask_question(query: str):

    if not query:
        print("Please enter a question!")
        return

    # Retrieve relevant chunks
    results = vector_store.similarity_search_with_score(
        query,
        k=TOP_K
    )

    relevant_texts = []

    print("\nRetrieved chunks:")

    for index, (doc, score) in enumerate(results, start=1):

        print(f"\nChunk {index}")
        print(f"Score: {score:.4f}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content: {doc.page_content[:100]}...")

        if score >= SCORE_THRESHOLD:
            relevant_texts.append(doc.page_content)

    # Check relevant documents
    if not relevant_texts:
        print("\nAnswer: I don't know.")
        return

    context = "\n\n".join(relevant_texts)

    # Generate answer
    messages = [
        {
            "role": "system",
            "content": """
            You are a helpful Cashfeet shop assistant.
            Answer only from the provided context.
            Never invent information.
            If the answer is unavailable, say I don't know.
            """
        },
        {
            "role": "user",
            "content": f"""
            Context:
            {context}

            Question:
            {query}
            """
        }
    ]

    response = llm.invoke(messages)

    print("\n========== ANSWER ==========")
    print(response.content)
    print("============================")


# =====================================
# 8. MAIN FUNCTION
# =====================================

def main():

    print("\n================================")
    print("       PDF BASED RAG SYSTEM")
    print("================================")

    print("Type exit to stop.")

    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        ask_question(query)


# =====================================
# 9. EXECUTION
# =====================================

if __name__ == "__main__":
    main()
