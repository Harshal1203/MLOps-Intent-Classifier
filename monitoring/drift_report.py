"""
Drift detection using Evidently AI.
Compares recent prediction inputs against the training baseline.
Triggers a retrain if drift exceeds the configured threshold.
"""

import os
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
from evidently.metrics import DatasetDriftMetric
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_recent_predictions(engine, hours: int = 24) -> pd.DataFrame:
    """Pull recent prediction logs from PostgreSQL."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    query = f"""
        SELECT text, predicted_intent, confidence, created_at
        FROM prediction_logs
        WHERE created_at >= '{cutoff.isoformat()}'
        ORDER BY created_at DESC
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Fetched {len(df)} recent predictions (last {hours}h)")
    return df


def load_reference_data(config: dict) -> pd.DataFrame:
    """Load training data as the drift reference baseline."""
    ref_path = Path(config["data"]["processed_path"]) / "train.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference data not found at {ref_path}. Run preprocess.py first.")
    df = pd.read_csv(ref_path)
    logger.info(f"Loaded {len(df)} reference samples")
    return df[["text", "intent"]].rename(columns={"intent": "predicted_intent"})


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, output_dir: str = "reports/"):
    """Generate Evidently drift report and return drift score."""
    Path(output_dir).mkdir(exist_ok=True)

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(),
    ])

    report.run(reference_data=reference[["text"]], current_data=current[["text"]])

    # Save HTML report
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"drift_report_{ts}.html"
    report.save_html(str(report_path))
    logger.info(f"Drift report saved: {report_path}")

    # Extract drift score
    result = report.as_dict()
    drift_score = result["metrics"][0]["result"]["dataset_drift_score"]
    logger.info(f"Dataset drift score: {drift_score:.4f}")
    return drift_score, str(report_path)


def trigger_retrain():
    """Placeholder — in production this would call the Kubeflow pipeline or GitHub Actions API."""
    logger.warning("🔁 DRIFT DETECTED — Triggering retrain pipeline...")
    # TODO: POST to Kubeflow endpoint or GitHub Actions workflow_dispatch API
    # Example:
    # import requests
    # requests.post(
    #     f"https://api.github.com/repos/{REPO}/actions/workflows/ci.yml/dispatches",
    #     headers={"Authorization": f"token {GITHUB_TOKEN}"},
    #     json={"ref": "main"},
    # )


def main():
    parser = argparse.ArgumentParser(description="Run drift detection report")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--hours",   type=int, default=24, help="Window of recent predictions to check")
    parser.add_argument("--dry-run", action="store_true",  help="Don't trigger retrain even if drift detected")
    args = parser.parse_args()

    config = load_config(args.config)
    threshold = config["monitoring"]["drift_threshold"]

    db_engine = create_engine(os.getenv("DATABASE_URL", config["database"]["url"]))

    # ── Fetch data ────────────────────────────────────────────────────────────
    current_data   = fetch_recent_predictions(db_engine, hours=args.hours)
    reference_data = load_reference_data(config)

    if len(current_data) < 50:
        logger.info("Not enough recent predictions to run drift detection (need >= 50). Skipping.")
        return

    # ── Run report ────────────────────────────────────────────────────────────
    drift_score, report_path = run_drift_report(reference_data, current_data)

    # ── Decision ──────────────────────────────────────────────────────────────
    if drift_score > threshold:
        logger.warning(f"⚠️  Drift score {drift_score:.4f} exceeds threshold {threshold}")
        if not args.dry_run and config["monitoring"]["retrain_on_drift"]:
            trigger_retrain()
        else:
            logger.info("[DRY RUN] Retrain would have been triggered")
    else:
        logger.info(f"✅ Drift score {drift_score:.4f} within acceptable range (threshold: {threshold})")


if __name__ == "__main__":
    main()
