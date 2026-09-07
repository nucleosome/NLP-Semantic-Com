"""Offline ingestion: paper + code comments → embeddings → local FAISS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CODE_SOURCES,
    CORPUS_PATH,
    EMBED_MODEL,
    EMBED_PROVIDER,
    FAISS_INDEX_PATH,
)
from rag_agent.providers import get_embeddings


def _load_documents() -> List[Document]:
    docs: List[Document] = []

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {CORPUS_PATH}. "
            "Add Xie et al. 2021 text to rag_agent/data/xie2021.txt."
        )

    paper = TextLoader(str(CORPUS_PATH), encoding="utf-8").load()
    for doc in paper:
        doc.metadata["source"] = CORPUS_PATH.name
        doc.metadata["kind"] = "paper"
    docs.extend(paper)

    repo_root = Path(__file__).resolve().parent.parent
    for path in CODE_SOURCES:
        path = Path(path)
        if not path.exists():
            continue
        loaded = TextLoader(str(path), encoding="utf-8").load()
        try:
            rel = str(path.relative_to(repo_root))
        except ValueError:
            rel = path.name
        for doc in loaded:
            doc.metadata["source"] = rel
            doc.metadata["kind"] = "code"
        docs.extend(loaded)

    if not docs:
        raise RuntimeError("No documents loaded for ingestion.")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_index(chunks: List[Document], persist: bool = True) -> FAISS:
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    if persist:
        Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        store.save_local(FAISS_INDEX_PATH)
    return store


def ingest(persist: bool = True) -> FAISS:
    docs = _load_documents()
    chunks = split_documents(docs)
    print(
        f"Loaded {len(docs)} document(s), split into {len(chunks)} chunk(s). "
        f"embed_provider={EMBED_PROVIDER} model={EMBED_MODEL}"
    )
    store = build_index(chunks, persist=persist)
    if persist:
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    return store


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the local FAISS index for the DeepSC RAG agent."
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Build the index in memory only (do not write to disk).",
    )
    args = parser.parse_args(argv)
    ingest(persist=not args.no_persist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
