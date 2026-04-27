"""
Evaluation script — runs the model on the test set, generates reports,
checks promotion thresholds, and logs everything to MLflow.
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from datasets import load_from_disk
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_from_registry(config: dict, stage: str = "Staging"):
    """Load model from MLflow Model Registry by stage."""
    model_name = config["mlflow"]["model_name"]
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading model from registry: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model


def run_inference(model, dataset, tokenizer, batch_size: int = 64, device: str = "cpu"):
    """Run batched inference and return predictions + latencies."""
    all_preds = []
    all_labels = []
    latencies = []

    model = model.to(device)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        input_ids      = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels         = batch["labels"]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        latency_ms = (time.perf_counter() - t0) * 1000

        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels if isinstance(labels, list) else labels.tolist())
        latencies.append(latency_ms / len(preds))  # per-sample latency

    return np.array(all_preds), np.array(all_labels), latencies


def evaluate(config: dict, run_id: str = None):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ── Load test data ────────────────────────────────────────────────────────
    processed_path = Path(config["data"]["processed_path"]) / "tokenized"
    dataset = load_from_disk(str(processed_path))
    test_ds = dataset["test"]

    id2label_path = Path(config["data"]["processed_path"]) / "id2label.json"
    with open(id2label_path) as f:
        id2label = json.load(f)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model_from_registry(config, stage="Staging")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running inference on test set...")
    preds, labels, latencies = run_inference(
        model, test_ds, tokenizer,
        batch_size=config["training"]["batch_size"],
        device=device,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy    = accuracy_score(labels, preds)
    f1_macro    = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    p95_latency = np.percentile(latencies, 95)
    avg_latency = np.mean(latencies)

    logger.info(f"Test Accuracy:    {accuracy:.4f}")
    logger.info(f"F1 Macro:         {f1_macro:.4f}")
    logger.info(f"F1 Weighted:      {f1_weighted:.4f}")
    logger.info(f"Avg Latency (ms): {avg_latency:.2f}")
    logger.info(f"P95 Latency (ms): {p95_latency:.2f}")

    # ── Save classification report ────────────────────────────────────────────
    label_names = [id2label[str(i)] for i in range(len(id2label))]
    report = classification_report(labels, preds, target_names=label_names, zero_division=0)
    report_path = Path("reports/classification_report.txt")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_id=run_id, run_name="evaluation"):
        mlflow.log_metrics({
            "test_accuracy":    accuracy,
            "test_f1_macro":    f1_macro,
            "test_f1_weighted": f1_weighted,
            "avg_latency_ms":   avg_latency,
            "p95_latency_ms":   p95_latency,
        })
        mlflow.log_artifact(str(report_path))

    # ── Promotion check ───────────────────────────────────────────────────────
    prod_threshold     = config["promotion"]["production_accuracy_threshold"]
    max_latency_ms     = config["promotion"]["max_latency_ms"]
    accuracy_pass      = accuracy     >= prod_threshold
    latency_pass       = p95_latency  <= max_latency_ms

    logger.info(f"\n{'='*50}")
    logger.info(f"PROMOTION GATE CHECK")
    logger.info(f"  Accuracy {accuracy:.4f} >= {prod_threshold}: {'✅ PASS' if accuracy_pass else '❌ FAIL'}")
    logger.info(f"  P95 Latency {p95_latency:.1f}ms <= {max_latency_ms}ms: {'✅ PASS' if latency_pass else '❌ FAIL'}")

    if accuracy_pass and latency_pass:
        logger.info("🎉 Model PASSES all gates → ready to promote to Production")
        return True
    else:
        logger.warning("🚫 Model FAILS promotion gates")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate intent classifier")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--run-id",  default=None, help="MLflow run ID to attach metrics to")
    args = parser.parse_args()

    config = load_config(args.config)
    passed = evaluate(config, run_id=args.run_id)
    exit(0 if passed else 1)


if __name__ == "__main__":
    main()
