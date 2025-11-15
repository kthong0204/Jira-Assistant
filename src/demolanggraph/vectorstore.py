from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .settings import SETTINGS, Settings


def save_vectorstore(
    documents: Iterable[Document],
    embeddings,
    *,
    path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Create and persist a Chroma vector store out of the provided documents."""

    cfg = settings or SETTINGS
    target_path = path or cfg.vector_store_path
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    store = Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        persist_directory=str(target_path),
    )
    # langchain_chroma persists automatically when using a persistent client,
    # but calling the underlying client's persist (if available) ensures data
    # is flushed for older chromadb versions.
    client = getattr(store, "_client", None)
    if client and hasattr(client, "persist"):
        client.persist()
    return target_path


def load_vectorstore(embeddings, path: Path | None = None) -> Chroma:
    """Load a Chroma store from disk."""

    target_path = path or SETTINGS.vector_store_path
    if not target_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {target_path}. Run `demo-rag ingest` first."
        )
    return Chroma(
        persist_directory=str(target_path),
        embedding_function=embeddings,
    )


def build_retriever(
    embeddings,
    settings: Settings | None = None,
) -> Chroma:
    cfg = settings or SETTINGS
    store = load_vectorstore(embeddings, cfg.vector_store_path)
    return store
