"""
Week 1 — Baseline: TF-IDF + Logistic Regression intent classifier.
Logs all experiments to MLflow.

Usage:
    python src/baseline.py
    python src/baseline.py --run-name tfidf-v2 --C 10
"""

import argparse
import json
import pickle
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR       = Path("data/processed")
MODELS_DIR          = Path("models")
REPORTS_DIR         = Path("reports")
PROMOTION_THRESHOLD = 0.88
load_dotenv()

def load_split(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run preprocess.py first.")
    return pd.read_csv(path)


def print_eda(splits: dict):
    logger.info("\n" + "="*58)
    logger.info("  DATASET SUMMARY")
    logger.info("="*58)
    for name, df in splits.items():
        logger.info(f"  {name:<8}: {len(df):>6} samples  |  {df['intent_label'].nunique()} intents")
    train_df = splits["train"]
    counts = train_df["intent_label"].value_counts()
    logger.info(f"\n  Class balance (train):")
    logger.info(f"    min  samples/class : {counts.min()}")
    logger.info(f"    max  samples/class : {counts.max()}")
    logger.info(f"    mean samples/class : {counts.mean():.1f}")
    logger.info(f"\n  Examples:")
    for label in train_df["intent_label"].unique()[:4]:
        ex = train_df[train_df["intent_label"] == label]["text"].iloc[0]
        logger.info(f"    [{label:30s}] {ex}")
    logger.info("="*58 + "\n")


def build_pipeline(tfidf_params: dict, lr_params: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LogisticRegression(**lr_params)),
    ])


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LR baseline")
    parser.add_argument("--experiment", default="intent-classifier")
    parser.add_argument("--run-name",   default="tfidf-baseline-v1")
    parser.add_argument("--C",          type=float, default=5.0,  help="LR regularization")
    parser.add_argument("--max-features", type=int, default=50000, help="TF-IDF vocab size")
    parser.add_argument("--ngram-max",  type=int,   default=2,    help="Max ngram size (1=unigram, 2=bigram)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    splits = {s: load_split(s) for s in ["train", "val", "test"]}
    print_eda(splits)

    X_train = splits["train"]["text"].tolist()
    y_train = splits["train"]["intent_label"].tolist()
    X_val   = splits["val"]["text"].tolist()
    y_val   = splits["val"]["intent_label"].tolist()
    X_test  = splits["test"]["text"].tolist()
    y_test  = splits["test"]["intent_label"].tolist()

    tfidf_params = {
        "ngram_range":  (1, args.ngram_max),
        "max_features": args.max_features,
        "sublinear_tf": True,
        "min_df":       2,
    }
    lr_params = {
        "C":           args.C,
        "max_iter":    1000,
        "solver":      "lbfgs",
        "n_jobs":      -1,
    }

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        logger.info(f"MLflow run: {run.info.run_id}")

        # Log params
        mlflow.log_params({
            "model_type":    "tfidf+logreg",
            "ngram_range":   f"(1,{args.ngram_max})",
            "max_features":  args.max_features,
            "sublinear_tf":  True,
            "C":             args.C,
            "solver":        "lbfgs",
            "train_samples": len(X_train),
            "val_samples":   len(X_val),
            "test_samples":  len(X_test),
            "n_classes":     len(set(y_train)),
        })
        mlflow.set_tag("week", "1-baseline")

        # Train
        logger.info("Training TF-IDF + Logistic Regression...")
        t0       = time.perf_counter()
        pipeline = build_pipeline(tfidf_params, lr_params)
        pipeline.fit(X_train, y_train)
        train_s  = time.perf_counter() - t0
        logger.info(f"  Training time: {train_s:.2f}s")

        # Val metrics
        val_preds = pipeline.predict(X_val)
        val_acc   = accuracy_score(y_val, val_preds)
        val_f1    = f1_score(y_val, val_preds, average="macro", zero_division=0)
        mlflow.log_metrics({"val_accuracy": val_acc, "val_f1_macro": val_f1})
        logger.info(f"  Val  acc={val_acc:.4f}  f1={val_f1:.4f}")

        # Test metrics
        t1          = time.perf_counter()
        test_preds  = pipeline.predict(X_test)
        latency_ms  = (time.perf_counter() - t1) / len(X_test) * 1000
        test_acc    = accuracy_score(y_test, test_preds)
        test_f1     = f1_score(y_test, test_preds, average="macro",    zero_division=0)
        test_f1_w   = f1_score(y_test, test_preds, average="weighted", zero_division=0)
        mlflow.log_metrics({
            "test_accuracy":    test_acc,
            "test_f1_macro":    test_f1,
            "test_f1_weighted": test_f1_w,
            "train_time_s":     train_s,
            "avg_latency_ms":   latency_ms,
        })
        logger.info(f"  Test acc={test_acc:.4f}  f1_macro={test_f1:.4f}  latency={latency_ms:.3f}ms/sample")

        # Classification report
        REPORTS_DIR.mkdir(exist_ok=True)
        report     = classification_report(y_test, test_preds, zero_division=0)
        report_path = REPORTS_DIR / f"{args.run_name}_report.txt"
        report_path.write_text(
            f"Run: {args.run_name} | ID: {run.info.run_id}\n"
            f"Test acc={test_acc:.4f}  F1={test_f1:.4f}\n\n" + report
        )
        mlflow.log_artifact(str(report_path))
        logger.info(f"  Classification report -> {report_path}")

        # Save model
        MODELS_DIR.mkdir(exist_ok=True)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        model_path = MODELS_DIR / "baseline_latest.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        # Promotion gate
        mlflow.set_tag("promotion_eligible", str(test_acc >= PROMOTION_THRESHOLD))
        if test_acc >= PROMOTION_THRESHOLD:
            logger.info(f"  PROMOTION ELIGIBLE (acc {test_acc:.4f} >= {PROMOTION_THRESHOLD})")
        else:
            logger.info(f"  Below promotion threshold ({test_acc:.4f} < {PROMOTION_THRESHOLD}) — transformer should beat this")

        logger.info(f"\nMLflow UI: http://localhost:5000  (run: mlflow ui)")
        logger.info(f"Run ID: {run.info.run_id}")

    logger.info("Week 1 baseline complete.")


if __name__ == "__main__":
    main()
