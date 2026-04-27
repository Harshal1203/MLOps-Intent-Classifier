"""
FastAPI inference server for the intent classifier.
Loads the Production model from MLflow Registry and serves predictions.
Logs all predictions to PostgreSQL for monitoring/drift detection.
"""

import os
import json
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
import torch
import mlflow
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
import numpy as np
from sqlalchemy import create_engine, text

from schemas import PredictRequest, PredictResponse, HealthResponse, VersionResponse, IntentPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Globals (populated at startup) ────────────────────────────────────────────
model       = None
tokenizer   = None
id2label    = None
model_meta  = {}
db_engine   = None


def load_config(config_path: str = "../configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def init_model(config: dict):
    global model, tokenizer, id2label, model_meta

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]))

    model_name  = config["mlflow"]["model_name"]
    model_stage = os.getenv("MODEL_STAGE", config["serving"]["model_stage"])
    model_uri   = f"models:/{model_name}/{model_stage}"

    logger.info(f"Loading model: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    # Load label map
    id2label_path = Path(config["data"]["processed_path"]) / "id2label.json"
    with open(id2label_path) as f:
        raw = json.load(f)
        id2label = {int(k): v for k, v in raw.items()}

    # Fetch version info from registry
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[model_stage])
    if versions:
        v = versions[0]
        model_meta.update({
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
        })

    logger.info(f"✅ Model loaded — version {model_meta.get('version')} ({model_stage})")


def init_db(config: dict):
    global db_engine
    db_url = os.getenv("DATABASE_URL", config["database"]["url"])
    try:
        db_engine = create_engine(db_url)
        with db_engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    predicted_intent TEXT,
                    confidence FLOAT,
                    model_version TEXT,
                    latency_ms FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            conn.commit()
        logger.info("✅ Database connection established")
    except Exception as e:
        logger.warning(f"DB init failed (predictions won't be logged): {e}")
        db_engine = None


def log_prediction(text: str, intent: str, confidence: float, version: str, latency_ms: float):
    if db_engine is None:
        return
    try:
        with db_engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO prediction_logs (text, predicted_intent, confidence, model_version, latency_ms)
                VALUES (:text, :intent, :confidence, :version, :latency_ms)
            """), {"text": text, "intent": intent, "confidence": confidence,
                   "version": version, "latency_ms": latency_ms})
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    init_model(config)
    init_db(config)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Intent Classifier API",
    description="Production ML API for multi-class intent classification",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_name="intent-classifier",
        model_stage=model_meta.get("stage", "unknown"),
    )


@app.get("/version", response_model=VersionResponse)
def version():
    return VersionResponse(
        model_name="intent-classifier",
        model_version=model_meta.get("version", "unknown"),
        model_stage=model_meta.get("stage", "unknown"),
        run_id=model_meta.get("run_id"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    config = load_config()
    max_length = config["model"]["max_length"]

    t0 = time.perf_counter()

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probs  = torch.softmax(logits, dim=-1).numpy()

    latency_ms = (time.perf_counter() - t0) * 1000

    top_k_idx  = np.argsort(probs)[::-1][:5]
    top_k      = [
        IntentPrediction(intent=id2label[i], confidence=float(probs[i]))
        for i in top_k_idx
    ]
    best       = top_k[0]
    model_ver  = model_meta.get("version", "unknown")

    log_prediction(request.text, best.intent, best.confidence, model_ver, latency_ms)

    return PredictResponse(
        text=request.text,
        prediction=best,
        top_k=top_k,
        model_version=model_ver,
        latency_ms=round(latency_ms, 2),
    )
