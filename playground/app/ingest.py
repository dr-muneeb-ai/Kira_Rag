
def ingest_pdf(path: str | None = None) -> int:
    settings = get_settings()
    pdf_path = resolve_data_path(settings) if path is None else Path(path).expanduser().resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = PyPDFLoader(str(pdf_path)).load()

    if not documents:
        raise ValueError("PDF contains no readable pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks generated.")

    embeddings = get_embeddings(settings)

    qdrant_path = Path(settings.qdrant_path).expanduser()
    if not qdrant_path.is_absolute():
        qdrant_path = Path(__file__).resolve().parents[1] / qdrant_path

    qdrant_path.parent.mkdir(parents=True, exist_ok=True)

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=str(qdrant_path.resolve()),
        collection_name=settings.collection_name,
        force_recreate=True,
    )

    print(f"PDF: {pdf_path}")
    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Qdrant collection: {settings.collection_name}")

    return len(chunks)

