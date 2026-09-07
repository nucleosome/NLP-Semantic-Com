"""CLI: python -m rag_agent.run \"What is the DeepSC architecture?\""""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_agent.config import (
    DEFAULT_CHANNEL,
    DEFAULT_SNR_DB,
    EMBED_PROVIDER,
    FAISS_INDEX_PATH,
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_RETRIES,
)
from rag_agent.graph import ask


def _index_ready() -> bool:
    root = Path(FAISS_INDEX_PATH)
    return (root / "index.faiss").exists() or root.with_suffix(".faiss").exists() or root.exists() and any(root.glob("*"))


def _print_result(state: dict) -> None:
    print("\n" + "=" * 72)
    print("QUESTION")
    print("=" * 72)
    print(state.get("question", ""))

    query = state.get("retrieval_query")
    rewritten = state.get("rewritten_question")
    if rewritten and rewritten != state.get("question"):
        print("\nRewritten query:", rewritten)
    elif query and query != state.get("question"):
        print("\nRetrieval query:", query)

    sources = state.get("document_sources") or []
    if sources:
        print("\nRetrieved sources:", ", ".join(sources))

    print("\nGrade:", state.get("grade", "n/a"), "| retries:", state.get("retries", 0))

    print("\n" + "=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(state.get("answer") or "(no answer)")

    sim = state.get("channel_sim")
    if sim is None:
        return

    print("\n" + "=" * 72)
    print("DEEPSC CHANNEL SIMULATION")
    print("=" * 72)
    print(json.dumps(sim, indent=2, ensure_ascii=False))
    if sim.get("ok"):
        print(
            f"\nBLEU={sim.get('bleu_score')}  "
            f"SNR={sim.get('snr_db')} dB  "
            f"channel={sim.get('channel')}"
        )
        print("Tx :", sim.get("original"))
        print("Rx :", sim.get("reconstructed"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the DeepSC RAG agent (Xie et al. 2021 + channel sim tool)."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="User question. Example: What is the DeepSC architecture?",
    )
    parser.add_argument(
        "--no-channel-sim",
        action="store_true",
        help="Skip the DeepSC wireless-channel demonstration.",
    )
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR_DB, help="SNR in dB for channel_sim.")
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        choices=["AWGN", "Rayleigh", "Rician"],
        help="Wireless channel model for channel_sim.",
    )
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip()
    if not question:
        parser.error('Provide a question, e.g. python -m rag_agent.run "What is DeepSC?"')

    if not _index_ready():
        print(
            f"FAISS index not found at {FAISS_INDEX_PATH}.\n"
            "Build it once with (same embedding provider as runtime):\n"
            "  cp .env.example .env   # set GOOGLE_API_KEY for Gemini\n"
            "  python -m rag_agent.ingest",
            file=sys.stderr,
        )
        return 1

    print(
        f"llm={LLM_PROVIDER}/{LLM_MODEL}  embed={EMBED_PROVIDER}  "
        f"max_retries={MAX_RETRIES}  channel_sim={not args.no_channel_sim}"
    )
    state = ask(
        question,
        run_channel_sim=not args.no_channel_sim,
        snr_db=args.snr,
        channel=args.channel,
    )
    _print_result(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
