from __future__ import annotations

from typing import Sequence

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import get_embeddings
from .settings import SETTINGS, Settings
from .vectorstore import save_vectorstore


def load_documents(settings: Settings | None = None) -> Sequence[Document]:
    cfg = settings or SETTINGS
    cfg.ensure_directories()

    docs: list[Document] = []
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(cfg.data_dir),
            glob=pattern,
            show_progress=True,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs.extend(loader.load())
    if not docs:
        raise RuntimeError(
            f"No documents found under {cfg.data_dir}. Place .txt/.md files there."
        )
    return docs


def split_documents(
    documents: Sequence[Document],
    settings: Settings | None = None,
) -> Sequence[Document]:
    cfg = settings or SETTINGS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    return splitter.split_documents(documents)


def ingest(settings: Settings | None = None) -> str:
    cfg = settings or SETTINGS
    docs = split_documents(load_documents(cfg), cfg)
    embeddings = get_embeddings(cfg)
    vector_path = save_vectorstore(docs, embeddings, settings=cfg)
    return f"Vector store saved to {vector_path}"


def main() -> None:
    print(ingest())


if __name__ == "__main__":
    main()
