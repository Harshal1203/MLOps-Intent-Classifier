"""
Fine-tuning pipeline for DistilBERT intent classifier.
All experiments are tracked with MLflow — params, metrics, and model artifacts.
"""

import os
import logging
import argparse
from pathlib import Path

import yaml
import numpy as np
import mlflow
import mlflow.pytorch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 for Trainer evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}


def train(config: dict, run_name: str = None):
    # ── MLflow setup ──────────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log all config params
        mlflow.log_params({
            "base_model":     config["model"]["base_model"],
            "num_labels":     config["model"]["num_labels"],
            "max_length":     config["model"]["max_length"],
            "epochs":         config["training"]["epochs"],
            "batch_size":     config["training"]["batch_size"],
            "learning_rate":  config["training"]["learning_rate"],
            "warmup_steps":   config["training"]["warmup_steps"],
            "weight_decay":   config["training"]["weight_decay"],
        })

        # ── Load data ─────────────────────────────────────────────────────────
        processed_path = Path(config["data"]["processed_path"]) / "tokenized"
        logger.info(f"Loading tokenized dataset from {processed_path}")
        dataset = load_from_disk(str(processed_path))

        train_ds = dataset["train"]
        val_ds   = dataset["validation"]

        # ── Model ─────────────────────────────────────────────────────────────
        model_name  = config["model"]["base_model"]
        num_labels  = config["model"]["num_labels"]

        logger.info(f"Loading model: {model_name} ({num_labels} classes)")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

        # ── Training args ─────────────────────────────────────────────────────
        output_dir = config["training"]["output_dir"]
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config["training"]["epochs"],
            per_device_train_batch_size=config["training"]["batch_size"],
            per_device_eval_batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            warmup_steps=config["training"]["warmup_steps"],
            weight_decay=config["training"]["weight_decay"],
            fp16=config["training"]["fp16"] and torch.cuda.is_available(),
            evaluation_strategy="steps",
            eval_steps=config["training"]["eval_steps"],
            save_steps=config["training"]["save_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_dir="logs/",
            logging_steps=100,
            report_to="none",  # We handle MLflow logging manually
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # ── Train ─────────────────────────────────────────────────────────────
        logger.info("Starting training...")
        trainer.train()

        # ── Evaluate on val set ───────────────────────────────────────────────
        logger.info("Evaluating on validation set...")
        eval_results = trainer.evaluate()
        mlflow.log_metrics({
            "val_accuracy": eval_results["eval_accuracy"],
            "val_f1_macro": eval_results["eval_f1_macro"],
            "val_loss":     eval_results["eval_loss"],
        })
        logger.info(f"Val accuracy: {eval_results['eval_accuracy']:.4f} | F1: {eval_results['eval_f1_macro']:.4f}")

        # ── Log model artifact ────────────────────────────────────────────────
        logger.info("Logging model to MLflow...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=config["mlflow"]["model_name"],
        )

        # ── Check promotion threshold ─────────────────────────────────────────
        staging_threshold = config["promotion"]["staging_accuracy_threshold"]
        if eval_results["eval_accuracy"] >= staging_threshold:
            logger.info(f"✅ Accuracy {eval_results['eval_accuracy']:.4f} >= threshold {staging_threshold}. Model eligible for Staging.")
            mlflow.set_tag("promotion_eligible", "true")
        else:
            logger.warning(f"❌ Accuracy {eval_results['eval_accuracy']:.4f} < threshold {staging_threshold}. Model NOT eligible.")
            mlflow.set_tag("promotion_eligible", "false")

        mlflow.set_tag("run_name", run_name or "default")
        logger.info(f"✅ Training complete. Run ID: {run.info.run_id}")
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, run_name=args.run_name)


if __name__ == "__main__":
    main()
