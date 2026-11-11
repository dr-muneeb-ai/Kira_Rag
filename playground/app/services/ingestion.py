from pathlib import Path

from app.ingest import ingest_pdf


def ingest(path: str | None = None) -> int:
    """Index the configured PDF, or temporarily override DATA_PATH."""
    if path is None:
        return ingest_pdf()

    # The core settings are intentionally immutable/cached; a path override
    # is handled by the CLI/config instead of mutating global settings here.
    raise NotImplementedError(
        "Set DATA_PATH in playground/.env to select a different PDF, "
        "then call ingest_pdf()."
    )


def load(path: str | Path | None = None):
    """Load PDF pages without indexing them."""
    from langchain_community.document_loaders import PyPDFLoader

    from app.config import get_settings, resolve_data_path

    pdf_path = Path(path).expanduser() if path else resolve_data_path(get_settings())
    if not pdf_path.is_absolute():
        pdf_path = (Path.cwd() / pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return PyPDFLoader(str(pdf_path)).load()
