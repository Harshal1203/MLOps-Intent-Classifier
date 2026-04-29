"""
Preprocessing pipeline for CLINC150 downloaded via HuggingFace datasets.

Input  (data/raw/):
    train.json  — {"text": str, "intent": int}  one record per line
    val.json
    test.json
    id2label.json — {int_str: label_name, ...}

Output (data/processed/):
    train.csv, val.csv, test.csv  — columns: text, intent_id, intent_label
    label2id.json, id2label.json  — copied for downstream use
    stats.json                    — dataset statistics

Usage:
    python src/preprocess.py
    python src/preprocess.py --raw-dir data/raw --out-dir data/processed
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_json_split(path: Path, id2label: dict) -> pd.DataFrame:
    """
    Load a HuggingFace .to_json() file.
    Each line: {"text": "...", "intent": 42}
    Returns DataFrame with columns: text, intent_id, intent_label
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            intent_id = row["intent"]
            records.append({
                "text":         row["text"].strip(),
                "intent_id":    intent_id,
                "intent_label": id2label[str(intent_id)],
            })
    return pd.DataFrame(records)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic text cleaning."""
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() >= 3]
    df = df.dropna(subset=["text", "intent_label"])
    df = df.reset_index(drop=True)
    logger.info(f"  Cleaning: {before} -> {len(df)} rows ({before - len(df)} removed)")
    return df


def print_stats(splits: dict):
    logger.info("\n" + "="*55)
    logger.info("  DATASET STATISTICS")
    logger.info("="*55)
    for name, df in splits.items():
        logger.info(f"  {name:<8}: {len(df):>5} samples | {df['intent_label'].nunique():>3} intents")
    all_df = pd.concat(splits.values())
    counts = all_df["intent_label"].value_counts()
    logger.info(f"\n  Samples per intent:")
    logger.info(f"    min  : {counts.min()}")
    logger.info(f"    max  : {counts.max()}")
    logger.info(f"    mean : {counts.mean():.1f}")
    logger.info(f"\n  Sample utterances:")
    for label in list(splits["train"]["intent_label"].unique())[:4]:
        sample = splits["train"][splits["train"]["intent_label"] == label]["text"].iloc[0]
        logger.info(f"    [{label}] {sample}")
    logger.info("="*55 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw",       help="Directory with train/val/test .json files")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for processed CSVs")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load label map
    id2label_path = raw_dir / "id2label.json"
    if not id2label_path.exists():
        raise FileNotFoundError(
            f"{id2label_path} not found.\n"
            "Run: python src/save_label_map.py  (requires internet + HF datasets)"
        )
    with open(id2label_path) as f:
        id2label = json.load(f)  # keys are strings e.g. "0", "1", ...
    logger.info(f"Loaded label map: {len(id2label)} intents")

    # Load splits
    splits_raw = {}
    for split in ["train", "val", "test"]:
        path = raw_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        logger.info(f"Loading {path}...")
        splits_raw[split] = load_json_split(path, id2label)

    # Clean
    logger.info("Cleaning splits...")
    splits_clean = {name: clean(df) for name, df in splits_raw.items()}

    # Stats
    print_stats(splits_clean)

    # Save CSVs
    stats = {}
    for name, df in splits_clean.items():
        out_path = out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {out_path}  ({len(df)} rows)")
        stats[name] = {
            "n_samples": len(df),
            "n_intents": df["intent_label"].nunique(),
        }

    # Copy label maps to processed/
    for fname in ["label2id.json", "id2label.json"]:
        src = raw_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    # Save stats.json
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Preprocessing complete -> data/processed/")


if __name__ == "__main__":
    main()
