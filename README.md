# DeepSC: Deep Learning-Enabled Semantic Communication System

A reproduction and deployment of **DeepSC** — a Transformer-based semantic communication system that applies NLP techniques to wireless transmission, enabling more efficient text transmission by preserving *meaning* rather than raw bits.

> **Reference:** H. Xie, Z. Qin, G. Y. Li, and B.-H. Juang, "Deep Learning Enabled Semantic Communication Systems," *IEEE Transactions on Signal Processing*, vol. 69, pp. 2663–2675, 2021. [DOI: 10.1109/TSP.2021.3071210](https://doi.org/10.1109/TSP.2021.3071210)

---

## Overview

Traditional communication systems encode text as bits and optimize for bit-error rate. DeepSC takes a fundamentally different approach: it uses a **Transformer encoder-decoder** to compress the *semantic content* of a sentence directly into a low-dimensional signal, transmits it over a noisy wireless channel, and reconstructs the original meaning at the receiver — even under severe noise conditions.

This project reproduces the DeepSC architecture and wraps it into a production-ready ML pipeline:

```
Notebook → Python Package → FastAPI Service → pytest → Docker → AWS ECS → GitHub Actions CI/CD
```

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────────────────┐
│  Semantic Encoder (Transformer, 4L/8H)  │
│  128-dim → captures sentence meaning    │
└────────────────────┬────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  Channel Encoder      │
         │  128 → 16 dims        │
         │  (power constrained)  │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Wireless Channel     │
         │  AWGN / Rayleigh /    │
         │  Rician (SNR: 0-18dB) │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Channel Decoder      │
         │  16 → 128 dims        │
         └───────────┬───────────┘
                     │
┌────────────────────▼────────────────────┐
│  Semantic Decoder (Transformer, 4L/8H)  │
│  Cross-attention → reconstructed text   │
└─────────────────────────────────────────┘
```

**Training objective:** Cross-entropy loss + Mutual Information (MINE) regularization to maximize information preserved through the channel.

**Dataset:** [Europarl v7](https://www.statmt.org/europarl/) English corpus

---

## Results

BLEU score vs. channel SNR on AWGN channel:

| SNR (dB) | BLEU Score |
|----------|-----------|
| 0        | ~0.55     |
| 6        | ~0.75     |
| 12       | ~0.85     |
| 18       | ~0.87     |

DeepSC maintains high semantic fidelity even at low SNR conditions where traditional systems fail.

---

## Project Structure

```
├── src/
│   ├── transceiver.py      # DeepSC Transformer architecture (Encoder, Channel, Decoder)
│   ├── mutual_info.py      # MINE mutual information estimator (auxiliary loss)
│   └── inference.py        # Inference wrapper (DeepSCInference class)
├── api/
│   ├── main.py             # FastAPI service (POST /predict, GET /health, GET /metrics)
│   ├── schemas.py          # Pydantic request/response models
│   └── model_loader.py     # Singleton model loader
├── notebooks/
│   └── 01_demo.ipynb       # Interactive demo: BLEU vs SNR plots, sentence reconstruction
├── tests/
│   ├── test_channel.py     # Unit tests: AWGN/Rayleigh/Rician channel models
│   ├── test_inference.py   # Unit tests: tokenization, encoding, decoding pipeline
│   └── test_api.py         # Integration tests: FastAPI endpoints via TestClient
├── main.py                 # Training script
├── performance.py          # BLEU evaluation across SNR levels
├── dataset.py              # EurDataset PyTorch Dataset + collate_data
├── utils.py                # Channels, loss, masks, greedy decoding
├── preprocess_text.py      # Europarl corpus preprocessing + vocab building
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess data
```bash
# Download Europarl v7 English corpus first
python preprocess_text.py
```

### 3. Train
```bash
python main.py --vocab-file data/vocab.json --checkpoint saved_models/ --channel AWGN
```

### 4. Evaluate
```bash
python performance.py --vocab-file data/vocab.json --checkpoint saved_models/deepsc_AWGN.pth
```

### 5. Run API server
```bash
uvicorn api.main:app --reload
# POST http://localhost:8000/predict
```

### 6. Docker
```bash
docker build -t deepsc-api .
docker run -p 8000:8000 -v $(pwd)/saved_models:/app/saved_models deepsc-api
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The parliament approved the budget.", "channel": "AWGN", "snr": 10}'
```

```json
{
  "reconstructed": "The parliament approved the budget.",
  "channel": "AWGN",
  "snr": 10.0
}
```

---

## CI/CD Pipeline

GitHub Actions automatically:
1. **Test** — runs `pytest tests/` on every push
2. **Build** — builds Docker image and pushes to AWS ECR (on `main` branch)
3. **Deploy** — triggers ECS Fargate rolling update

---

## Reference

```bibtex
@article{xie2021deepsc,
  author  = {Huiqiang Xie and Zhijin Qin and Geoffrey Ye Li and Biing-Hwang Juang},
  title   = {Deep Learning Enabled Semantic Communication Systems},
  journal = {IEEE Transactions on Signal Processing},
  volume  = {69},
  pages   = {2663--2675},
  year    = {2021},
  doi     = {10.1109/TSP.2021.3071210}
}
```

---

## Updates

### RAG research agent (LangGraph + FAISS)

Added a research Q&A agent over **Xie et al. 2021** (and this repo's source comments). Default LLM/embeddings are **Gemini** (free Google AI Studio key). OpenAI and Anthropic remain optional via `RAG_LLM_PROVIDER`. After answering, it can optionally pass a sentence through the trained DeepSC model (`channel_sim` LangChain Tool) to show BLEU under AWGN / Rayleigh / Rician.

**New layout**

```
├── rag_agent/
│   ├── config.py           # API keys, LLM / embedding models, FAISS path
│   ├── ingest.py           # Chunk Xie 2021 + source comments → FAISS
│   ├── nodes.py            # retrieve / grade / rewrite / generate / channel_sim
│   ├── tools.py            # LangChain Tool wrapping DeepSC + greedy_decode_L
│   ├── graph.py            # LangGraph StateGraph + retry policy
│   ├── run.py              # CLI: python -m rag_agent.run "your question"
│   └── data/README.md      # How to add local xie2021.txt (gitignored)
├── tests/test_rag_agent.py # RAG graph routing + mocked channel_sim tool
└── .env.example            # Template for GOOGLE_API_KEY / provider overrides
```

**Flow**

```
User question
   │
   ▼
LangGraph Agent (rag_agent/graph.py)
   │
   ▼
[retrieve]  →  FAISS top-k over local xie2021.txt + src/*.py comments
   │
   ▼
[grade]     →  LLM (Gemini / OpenAI / Anthropic): retrieved docs relevant?
   │          │
   ▼          ▼
[generate]  [rewrite → retrieve]   ← at most RAG_MAX_RETRIES (default 2)
   │
   ▼
[channel_sim] (optional) → DeepSC + greedy_decode_L → reconstructed text + BLEU
   │
   ▼
Final answer
```

**Setup**

```bash
cp .env.example .env   # set GOOGLE_API_KEY (https://aistudio.google.com/apikey)
pip install -r requirements.txt

# Add paper text yourself (not in the repo), e.g. from arXiv:2006.10685
# → rag_agent/data/xie2021.txt   (see rag_agent/data/README.md)

# Build the local FAISS index once (must match RAG_EMBED_PROVIDER)
python -m rag_agent.ingest
```

Changing embedding provider or model requires rebuilding the FAISS index. Both `rag_agent/data/xie2021.txt` and `rag_agent/data/faiss_index/` are gitignored — supply the corpus and rebuild the index locally after clone.

**Ask a question**

```bash
python -m rag_agent.run "What is the DeepSC architecture?"
python -m rag_agent.run --snr 8 --channel Rayleigh "How does DeepSC handle low SNR?"
python -m rag_agent.run --no-channel-sim "What is sentence similarity vs BLEU?"
```

`channel_sim` loads the same checkpoint as the API (`DEEPSC_CHECKPOINT`, `DEEPSC_VOCAB`). If those files are missing, the agent still answers from the paper and prints a clear skip message for the wireless demo. DeepSC's vocab is Europarl (max 30 tokens); OOV technical terms become `<UNK>` and can lower BLEU — that is expected.
