"""LangChain tools wrapping the trained DeepSC transceiver."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import torch
from langchain_core.tools import tool

SPECIAL_TOKENS = {"<PAD>", "<START>", "<END>", "<UNK>"}
MAX_SIM_LEN = 30


def _strip_special_tokens(text: str) -> str:
    words = [w for w in text.split() if w and w not in SPECIAL_TOKENS]
    return " ".join(words)


def _bleu(reference: str, hypothesis: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    ref = _strip_special_tokens(reference.lower()).split()
    hyp = _strip_special_tokens(hypothesis.lower()).split()
    if not ref or not hyp:
        return 0.0
    return float(
        sentence_bleu(
            [ref],
            hyp,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def _truncate_for_deepsc(text: str, max_words: int = MAX_SIM_LEN - 2) -> str:
    """DeepSC was trained on 4–30 word Europarl sentences."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    words = cleaned.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


def _load_deepsc():
    """Load inference wrapper + decode helpers. Isolated for easy mocking."""
    from api.model_loader import get_model
    from utils import SNR_to_noise, greedy_decode_L

    return get_model(), greedy_decode_L, SNR_to_noise


def run_channel_sim(
    text: str,
    snr_db: float = 12.0,
    channel: str = "AWGN",
    *,
    inference=None,
    decode_fn=None,
    snr_to_noise=None,
) -> Dict[str, Any]:
    """Run text through DeepSC + simulated wireless channel.

    Uses ``api.model_loader.get_model`` and ``utils.greedy_decode_L`` so the
    call returns reconstructed text, BLEU, and Tx/Rx signal power.

    ``inference`` / ``decode_fn`` / ``snr_to_noise`` are injectable for tests.
    """
    snippet = _truncate_for_deepsc(text)
    if not snippet:
        return {
            "ok": False,
            "error": "Empty text after truncation.",
            "original": text,
            "snr_db": snr_db,
            "channel": channel,
        }

    try:
        if inference is None or decode_fn is None or snr_to_noise is None:
            loaded_inf, loaded_decode, loaded_snr = _load_deepsc()
            inference = inference or loaded_inf
            decode_fn = decode_fn or loaded_decode
            snr_to_noise = snr_to_noise or loaded_snr
    except Exception as exc:
        return {
            "ok": False,
            "error": (
                "DeepSC checkpoint or vocab not available "
                f"({exc}). Train the model or set DEEPSC_CHECKPOINT / DEEPSC_VOCAB."
            ),
            "original": snippet,
            "snr_db": snr_db,
            "channel": channel,
        }

    wrapper = inference
    wrapper.channel = channel
    wrapper.n_var = snr_to_noise(snr_db)
    src = wrapper._tokenize(snippet)
    if src.size(1) > MAX_SIM_LEN:
        src = src[:, :MAX_SIM_LEN]

    try:
        with torch.no_grad():
            outputs, tx_sig, rx_sig = decode_fn(
                wrapper.model,
                src,
                wrapper.n_var,
                MAX_SIM_LEN,
                wrapper.pad_idx,
                wrapper.start_idx,
                channel,
            )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"DeepSC forward pass failed: {exc}",
            "original": snippet,
            "snr_db": snr_db,
            "channel": channel,
        }

    reconstructed = _strip_special_tokens(
        wrapper.seq2text.sequence_to_text(outputs[0].tolist())
    )
    tx_power = float(torch.mean(tx_sig ** 2).cpu())
    rx_power = float(torch.mean(rx_sig ** 2).cpu())

    return {
        "ok": True,
        "original": snippet,
        "reconstructed": reconstructed,
        "bleu_score": round(_bleu(snippet, reconstructed), 4),
        "snr_db": snr_db,
        "channel": channel,
        "tx_power": round(tx_power, 6),
        "rx_power": round(rx_power, 6),
        "note": (
            "DeepSC vocab is Europarl-based (lowercase, max 30 tokens). "
            "OOV technical terms become <UNK> and can lower BLEU."
        ),
    }


@tool
def channel_sim(text: str, snr_db: float = 12.0, channel: str = "AWGN") -> str:
    """Pass text through the trained DeepSC semantic communication model
    over a simulated wireless channel (AWGN / Rayleigh / Rician).

    Returns reconstructed text plus a BLEU score that measures how well
    meaning survived the noisy channel. Use this when the user asks what
    DeepSC would do to a sentence, or to demonstrate semantic transmission.
    """
    result = run_channel_sim(text=text, snr_db=snr_db, channel=channel)
    return json.dumps(result, ensure_ascii=False)


def snippet_for_sim(answer: str) -> Optional[str]:
    """Pick a short sentence from the generated answer for DeepSC demo."""
    if not answer:
        return None
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    candidate = parts[0] if parts else answer
    return _truncate_for_deepsc(candidate)
