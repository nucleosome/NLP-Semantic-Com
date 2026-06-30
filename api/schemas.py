from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="The parliament approved the budget.")
    channel: Literal["AWGN", "Rayleigh", "Rician"] = "AWGN"
    snr: float = Field(10.0, ge=0.0, le=20.0, description="Signal-to-noise ratio in dB")


class PredictResponse(BaseModel):
    reconstructed: str
    channel: str
    snr: float


class MetricsResponse(BaseModel):
    bleu_scores: list[float]
    snr_levels: list[int]
    description: str
