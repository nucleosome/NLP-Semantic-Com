import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = "The parliament approved the budget."
    mock_model.channel = "AWGN"
    mock_model.n_var = 0.1

    with patch("api.model_loader.get_model", return_value=mock_model):
        from api.main import app
        yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_success(client):
    resp = client.post("/predict", json={
        "text": "The parliament approved the budget.",
        "channel": "AWGN",
        "snr": 10.0
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "reconstructed" in data
    assert data["channel"] == "AWGN"
    assert data["snr"] == 10.0


def test_predict_invalid_channel(client):
    resp = client.post("/predict", json={
        "text": "hello",
        "channel": "InvalidChannel",
        "snr": 10.0
    })
    assert resp.status_code == 422


def test_predict_empty_text(client):
    resp = client.post("/predict", json={"text": "", "channel": "AWGN", "snr": 10.0})
    assert resp.status_code == 422


def test_predict_snr_out_of_range(client):
    resp = client.post("/predict", json={"text": "hello", "channel": "AWGN", "snr": 999})
    assert resp.status_code == 422


def test_metrics_not_found(client):
    with patch("api.main.np.load", side_effect=FileNotFoundError):
        resp = client.get("/metrics")
    assert resp.status_code == 404


def test_metrics_success(client):
    fake_bleu = np.array([0.55, 0.70, 0.80, 0.85, 0.87])
    with patch("api.main.np.load", return_value=fake_bleu):
        resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["bleu_scores"]) == 5
    assert "snr_levels" in data
