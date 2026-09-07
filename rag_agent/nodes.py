"""LangGraph node functions: retrieve / grade / rewrite / generate / channel_sim."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

from rag_agent.config import FAISS_INDEX_PATH, TOP_K
from rag_agent.providers import get_embeddings, get_llm
from rag_agent.tools import run_channel_sim, snippet_for_sim


@lru_cache(maxsize=1)
def get_retriever():
    embeddings = get_embeddings()
    store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store.as_retriever(search_kwargs={"k": TOP_K})


def _format_docs(docs) -> str:
    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] source={source}\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("rewritten_question") or state["question"]
    docs = get_retriever().invoke(query)
    return {
        "documents": [d.page_content for d in docs],
        "document_sources": [d.metadata.get("source", "unknown") for d in docs],
        "formatted_context": _format_docs(docs),
        "retrieval_query": query,
    }


def grade(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_llm(temperature=0.0)
    context = state.get("formatted_context") or "\n\n".join(state.get("documents") or [])
    prompt = (
        "You are a grader for a retrieval-augmented QA system about DeepSC "
        "(Xie et al., IEEE TSP 2021, deep learning enabled semantic communication).\n"
        "Decide if the retrieved documents contain enough information to answer "
        "the user question. Reply with exactly one token: relevant OR not_relevant.\n\n"
        f"Question: {state['question']}\n\n"
        f"Documents:\n{context}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    text = (response.content or "").strip().lower().replace("-", "_")
    compact = text.replace(" ", "")
    if "not_relevant" in compact or compact.startswith("no") or "irrelevant" in compact:
        label = "not_relevant"
    else:
        label = "relevant"
    return {"grade": label}


def rewrite(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_llm(temperature=0.2)
    prompt = (
        "Rewrite the user question so it is a better search query over the DeepSC "
        "paper (Xie et al. 2021) and the reproduction codebase. Keep it concise. "
        "Focus on architecture, channel model, loss (cross-entropy + MINE), BLEU / "
        "sentence similarity, transfer learning, or implementation details as relevant. "
        "Return only the rewritten query.\n\n"
        f"Original question: {state['question']}\n"
        f"Previous search query: {state.get('retrieval_query', state['question'])}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten = (response.content or state["question"]).strip().strip('"')
    retries = int(state.get("retries") or 0) + 1
    return {"rewritten_question": rewritten, "retries": retries}


def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_llm(temperature=0.2)
    context = state.get("formatted_context") or "\n\n".join(state.get("documents") or [])
    messages = [
        SystemMessage(
            content=(
                "You are an expert on DeepSC, the Transformer-based semantic "
                "communication system from Xie, Qin, Li, and Juang, IEEE Transactions "
                "on Signal Processing, 2021 (arXiv:2006.10685). You also know this "
                "repository's reproduction: src/transceiver.py, utils.py Channels, "
                "MINE mutual information in src/mutual_info.py, and the FastAPI wrapper.\n"
                "Answer using the retrieved context. Cite sources like [1] or file names. "
                "If the context is insufficient, say what is missing. Be technically precise."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\n"
                f"Retrieved context:\n{context}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"answer": (response.content or "").strip()}


def channel_sim_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("run_channel_sim", True):
        return {"channel_sim": None}

    text = snippet_for_sim(state.get("answer") or "")
    if not text:
        return {"channel_sim": {"ok": False, "error": "No answer text to simulate."}}

    snr_db = float(state.get("snr_db", 12.0))
    channel = str(state.get("channel", "AWGN"))
    return {"channel_sim": run_channel_sim(text, snr_db=snr_db, channel=channel)}
