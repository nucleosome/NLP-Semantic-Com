"""API keys, model names, and FAISS paths for the DeepSC RAG agent."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at import time
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = Path(__file__).resolve().parent
DATA_DIR = RAG_DIR / "data"

if load_dotenv is not None:
    # Prefer project .env over stale shell exports (e.g. old RAG_LLM_MODEL).
    load_dotenv(REPO_ROOT / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# openai | anthropic | gemini
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "gemini").lower()
# Embedding provider; Anthropic has no embeddings — defaults to openai if LLM is anthropic.
_raw_embed = os.getenv("RAG_EMBED_PROVIDER")
if _raw_embed:
    EMBED_PROVIDER = _raw_embed.lower()
elif LLM_PROVIDER == "anthropic":
    EMBED_PROVIDER = "openai"
else:
    EMBED_PROVIDER = LLM_PROVIDER

_DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "gemini": "gemini-3.6-flash",
}
_DEFAULT_EMBED_MODELS = {
    "openai": "text-embedding-3-small",
    # text-embedding-004 retired; gemini-embedding-001 is the current text embed model
    "gemini": "models/gemini-embedding-001",
}

LLM_MODEL = os.getenv(
    "RAG_LLM_MODEL",
    _DEFAULT_LLM_MODELS.get(LLM_PROVIDER, "gemini-3.6-flash"),
)
EMBED_MODEL = os.getenv(
    "RAG_EMBED_MODEL",
    _DEFAULT_EMBED_MODELS.get(EMBED_PROVIDER, "models/gemini-embedding-001"),
)

CORPUS_PATH = DATA_DIR / "xie2021.txt"
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index")

TOP_K = int(os.getenv("RAG_TOP_K", "4"))
MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "2"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))

DEFAULT_SNR_DB = float(os.getenv("RAG_SNR_DB", "12.0"))
DEFAULT_CHANNEL = os.getenv("RAG_CHANNEL", "AWGN")

# Python sources ingested alongside the paper (architecture + training comments).
CODE_SOURCES = [
    REPO_ROOT / "src" / "transceiver.py",
    REPO_ROOT / "src" / "inference.py",
    REPO_ROOT / "src" / "mutual_info.py",
    REPO_ROOT / "api" / "main.py",
    REPO_ROOT / "api" / "model_loader.py",
    REPO_ROOT / "utils.py",
    REPO_ROOT / "main.py",
    REPO_ROOT / "performance.py",
]


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or put it in a .env file "
            "(see .env.example)."
        )
    return OPENAI_API_KEY


def require_google_key() -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Get a free key from Google AI Studio "
            "(https://aistudio.google.com/apikey) and put it in .env "
            "(see .env.example)."
        )
    return GOOGLE_API_KEY


def require_anthropic_key() -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it or put it in a .env file "
            "(see .env.example)."
        )
    return ANTHROPIC_API_KEY
