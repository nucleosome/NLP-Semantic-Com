import numpy as np
from fastapi import FastAPI, HTTPException
from api.schemas import PredictRequest, PredictResponse, MetricsResponse
from api.model_loader import get_model

app = FastAPI(
    title="DeepSC Semantic Communication API",
    description=(
        "Transformer-based semantic communication system. "
        "Reproduces Xie et al. 2021 (IEEE TSP). "
        "Transmits text meaning through noisy wireless channels."
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")

    model.channel = request.channel
    from utils import SNR_to_noise
    model.n_var = SNR_to_noise(request.snr)

    reconstructed = model.predict(request.text)
    return PredictResponse(
        reconstructed=reconstructed,
        channel=request.channel,
        snr=request.snr,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    try:
        bleu = np.load("saved_models/bleu.npy").tolist()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="BLEU results not found")

    snr_levels = list(range(0, len(bleu) * 2, 2))
    return MetricsResponse(
        bleu_scores=bleu,
        snr_levels=snr_levels,
        description="BLEU scores vs SNR (dB) on AWGN channel, Europarl test set",
    )
