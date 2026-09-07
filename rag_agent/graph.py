"""LangGraph StateGraph: retrieve → grade → rewrite/generate → channel_sim."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_agent.config import DEFAULT_CHANNEL, DEFAULT_SNR_DB, MAX_RETRIES
from rag_agent.nodes import channel_sim_node, generate, grade, retrieve, rewrite


class AgentState(TypedDict, total=False):
    question: str
    rewritten_question: str
    retrieval_query: str
    documents: List[str]
    document_sources: List[str]
    formatted_context: str
    grade: Literal["relevant", "not_relevant"]
    retries: int
    answer: str
    run_channel_sim: bool
    snr_db: float
    channel: str
    channel_sim: Optional[Dict[str, Any]]


def decide_next(state: AgentState) -> Literal["relevant", "rewrite", "give_up"]:
    """Conditional edge after grade: generate, rewrite+re-retrieve, or give up."""
    if state.get("grade") == "relevant":
        return "relevant"
    if int(state.get("retries") or 0) >= MAX_RETRIES:
        return "give_up"
    return "rewrite"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("channel_sim", channel_sim_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        decide_next,
        {
            "relevant": "generate",
            "rewrite": "rewrite",
            "give_up": "generate",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "channel_sim")
    workflow.add_edge("channel_sim", END)
    return workflow.compile()


def ask(
    question: str,
    run_channel_sim: bool = True,
    snr_db: float = DEFAULT_SNR_DB,
    channel: str = DEFAULT_CHANNEL,
    app=None,
) -> AgentState:
    """Run the compiled graph once and return the final state."""
    graph = app or build_graph()
    initial: AgentState = {
        "question": question,
        "retries": 0,
        "run_channel_sim": run_channel_sim,
        "snr_db": snr_db,
        "channel": channel,
    }
    return graph.invoke(initial)
