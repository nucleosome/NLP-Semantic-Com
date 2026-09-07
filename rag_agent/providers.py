"""LLM and embedding factories for openai / anthropic / gemini."""

from __future__ import annotations

from rag_agent.config import (
    ANTHROPIC_API_KEY,
    EMBED_MODEL,
    EMBED_PROVIDER,
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    require_anthropic_key,
    require_google_key,
    require_openai_key,
)


def get_embeddings():
    """Return the embedding model used for ingest and retrieval (must match)."""
    if EMBED_PROVIDER == "gemini":
        require_google_key()
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    if EMBED_PROVIDER == "openai":
        require_openai_key()
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    raise RuntimeError(
        f"Unsupported RAG_EMBED_PROVIDER={EMBED_PROVIDER!r}. "
        "Use 'gemini' or 'openai' (Anthropic has no embeddings API)."
    )


def get_llm(temperature: float = 0.0):
    """Return the chat model for grade / rewrite / generate."""
    if LLM_PROVIDER == "gemini":
        require_google_key()
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
        )
    if LLM_PROVIDER == "anthropic":
        require_anthropic_key()
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=LLM_MODEL,
            temperature=temperature,
            api_key=ANTHROPIC_API_KEY,
        )
    if LLM_PROVIDER == "openai":
        require_openai_key()
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
        )
    raise RuntimeError(
        f"Unsupported RAG_LLM_PROVIDER={LLM_PROVIDER!r}. "
        "Use 'gemini', 'openai', or 'anthropic'."
    )
