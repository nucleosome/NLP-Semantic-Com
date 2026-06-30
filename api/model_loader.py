import os
from functools import lru_cache
from src.inference import DeepSCInference

CHECKPOINT_PATH = os.getenv("DEEPSC_CHECKPOINT", "saved_models/deepsc_AWGN.pth")
VOCAB_PATH = os.getenv("DEEPSC_VOCAB", "data/vocab.json")
CHANNEL = os.getenv("DEEPSC_CHANNEL", "AWGN")
SNR = float(os.getenv("DEEPSC_SNR", "10.0"))


@lru_cache(maxsize=1)
def get_model() -> DeepSCInference:
    return DeepSCInference(CHECKPOINT_PATH, VOCAB_PATH, CHANNEL, SNR)
