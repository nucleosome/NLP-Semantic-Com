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
