"""
Tests for the FastAPI inference server.
Uses httpx TestClient — no running model needed (model is mocked).
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model_globals():
    """Patch the model, tokenizer, and label map used by the API."""
    mock_model = MagicMock()
    mock_logits = MagicMock()
    mock_logits.logits = MagicMock()

    import torch
    import numpy as np

    # Simulate a model that returns high confidence for class 0
    fake_logits = torch.zeros(1, 150)
    fake_logits[0][0] = 5.0  # class 0 gets high logit

    mock_model.return_value = MagicMock(logits=fake_logits)

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 128, dtype=torch.long),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
    }

    fake_id2label = {i: f"intent_{i}" for i in range(150)}

    with patch("api.main.model", mock_model), \
         patch("api.main.tokenizer", mock_tokenizer), \
         patch("api.main.id2label", fake_id2label), \
         patch("api.main.model_meta", {"version": "1", "stage": "Production", "run_id": "abc123"}), \
         patch("api.main.db_engine", None):  # Skip DB logging
        yield


def get_client():
    # Import here to avoid loading model at module level
    from api.main import app
    return TestClient(app)


def test_health_endpoint():
    with patch("api.main.model", MagicMock()), \
         patch("api.main.model_meta", {"stage": "Production"}):
        client = get_client()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


def test_predict_valid_input(mock_model_globals):
    client = get_client()
    response = client.post("/predict", json={"text": "What is the weather like today?"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data["prediction"]
    assert "intent" in data["prediction"]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0
    assert len(data["top_k"]) == 5
    assert data["latency_ms"] > 0


def test_predict_empty_text():
    client = get_client()
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_predict_missing_text():
    client = get_client()
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_version_endpoint():
    with patch("api.main.model_meta", {"version": "3", "stage": "Production", "run_id": "xyz"}):
        client = get_client()
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "3"
        assert data["model_stage"] == "Production"
