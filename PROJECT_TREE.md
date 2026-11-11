# Project tree

```text
claude/
├── README.md
├── pyproject.toml
├── .gitignore
└── playground/
    ├── README.md
    ├── pyproject.toml
    ├── .env.example
    ├── app/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── main.py
    │   ├── rag.py
    │   ├── ingest.py
    │   └── services/
    │       ├── chat.py
    │       └── ingestion.py
    └── code/
        ├── config.py
        ├── rag.py
        ├── rag-03.py
        ├── shop_handbook.md
        ├── shop_handbook.html
        └── shop_handbook.pdf
```

Generated environments (`.venv`, `.play`), Git internals (`.git`), Python bytecode,
local Qdrant data, and secret `.env` files are intentionally excluded.
