"""Unit tests for rag_agent that do not call external LLM APIs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from rag_agent.config import CORPUS_PATH, MAX_RETRIES, RAG_DIR
from rag_agent.graph import decide_next
from rag_agent.tools import _truncate_for_deepsc, channel_sim, run_channel_sim, snippet_for_sim


def test_corpus_covers_required_sections():
    if not CORPUS_PATH.exists():
        pytest.skip("rag_agent/data/xie2021.txt not present (user-supplied corpus)")
    text = CORPUS_PATH.read_text(encoding="utf-8").lower()
    for needle in (
        "abstract",
        "introduction",
        "system model",
        "deepsc",
        "transformer",
        "mutual information",
        "bleu",
        "sentence similarity",
        "awgn",
        "rayleigh",
        "transfer learning",
    ):
        assert needle in text, f"xie2021.txt missing section/topic: {needle}"


def test_decide_next_relevant_goes_to_generate():
    assert decide_next({"grade": "relevant", "retries": 0}) == "relevant"


def test_decide_next_rewrite_when_under_budget():
    assert decide_next({"grade": "not_relevant", "retries": 0}) == "rewrite"
    assert decide_next({"grade": "not_relevant", "retries": MAX_RETRIES - 1}) == "rewrite"


def test_decide_next_give_up_after_max_retries():
    assert decide_next({"grade": "not_relevant", "retries": MAX_RETRIES}) == "give_up"


def test_truncate_and_snippet_respect_deepsc_length():
    long_text = "word " * 80
    out = _truncate_for_deepsc(long_text)
    assert len(out.split()) <= 28
    snippet = snippet_for_sim("First sentence is short. Second is ignored.")
    assert snippet == "First sentence is short."


def test_channel_sim_missing_checkpoint_is_graceful():
    with patch("rag_agent.tools._load_deepsc", side_effect=FileNotFoundError("no ckpt")):
        result = run_channel_sim("the parliament approved the budget", snr_db=12.0)
    assert result["ok"] is False
    assert "not available" in result["error"].lower()


def _fake_wrapper(reconstructed: str, token_ids):
    wrapper = MagicMock()
    wrapper._tokenize.return_value = torch.tensor([token_ids])
    wrapper.pad_idx = 0
    wrapper.start_idx = 1
    wrapper.seq2text.sequence_to_text.return_value = reconstructed
    wrapper.model = MagicMock()
    return wrapper


def test_channel_sim_mocked_forward_returns_bleu():
    text = "it is an important step towards equal rights for all"
    wrapper = _fake_wrapper("<START> " + text, list(range(12)))
    fake_out = torch.tensor([list(range(12))])
    fake_tx = torch.ones(1, 12, 16)
    fake_rx = torch.ones(1, 12, 16) * 0.9

    result = run_channel_sim(
        text,
        snr_db=12.0,
        channel="AWGN",
        inference=wrapper,
        decode_fn=lambda *a, **k: (fake_out, fake_tx, fake_rx),
        snr_to_noise=lambda snr: 0.1,
    )

    assert result["ok"] is True
    assert result["reconstructed"] == text
    assert result["bleu_score"] == 1.0
    assert result["channel"] == "AWGN"
    assert result["tx_power"] > 0
    assert "rx_power" in result


def test_channel_sim_tool_json_with_mock():
    with patch(
        "rag_agent.tools.run_channel_sim",
        return_value={
            "ok": True,
            "original": "hello",
            "reconstructed": "hello",
            "bleu_score": 1.0,
            "snr_db": 10.0,
            "channel": "Rayleigh",
        },
    ):
        payload = channel_sim.invoke({"text": "hello", "snr_db": 10.0, "channel": "Rayleigh"})

    data = json.loads(payload)
    assert data["ok"] is True
    assert data["channel"] == "Rayleigh"


def test_ingest_split_does_not_need_openai():
    from langchain_core.documents import Document

    from rag_agent.ingest import split_documents

    docs = [
        Document(page_content="DeepSC " * 200, metadata={"source": "xie2021.txt"}),
        Document(page_content="class DeepSC:\n    pass\n", metadata={"source": "transceiver.py"}),
    ]
    chunks = split_documents(docs)
    assert len(chunks) >= 2
    assert all(c.page_content for c in chunks)


def test_get_llm_rejects_unknown_provider():
    with patch("rag_agent.providers.LLM_PROVIDER", "nope"):
        from rag_agent import providers

        try:
            providers.get_llm()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "Unsupported RAG_LLM_PROVIDER" in str(exc)


def test_get_embeddings_rejects_anthropic():
    with patch("rag_agent.providers.EMBED_PROVIDER", "anthropic"):
        from rag_agent import providers

        try:
            providers.get_embeddings()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "openai" in str(exc).lower() or "gemini" in str(exc).lower()


def test_rag_package_layout():
    expected = [
        "config.py",
        "ingest.py",
        "nodes.py",
        "tools.py",
        "graph.py",
        "run.py",
        "providers.py",
    ]
    for name in expected:
        assert (RAG_DIR / name).is_file()
    assert CORPUS_PATH.is_file()


def test_build_graph_compiles_without_api_calls():
    from rag_agent.graph import build_graph

    app = build_graph()
    assert app is not None
