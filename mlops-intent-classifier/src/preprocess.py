"""
Data preprocessing pipeline for CLINC150 intent classification.
Handles loading, cleaning, tokenization, and train/val/test splitting.
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_clinc150(config: dict) -> DatasetDict:
    """Load the CLINC150 dataset from HuggingFace Hub."""
    logger.info("Loading CLINC150 dataset...")
    dataset = load_dataset("clinc_oos", "plus")
    logger.info(f"Dataset loaded: {dataset}")
    return dataset


def build_label_map(dataset: DatasetDict) -> Tuple[dict, dict]:
    """Build integer <-> label string mappings."""
    labels = dataset["train"].features["intent"].names
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    logger.info(f"Found {len(label2id)} intent classes")
    return label2id, id2label


def tokenize_dataset(dataset: DatasetDict, config: dict) -> DatasetDict:
    """Tokenize text using the configured base model tokenizer."""
    model_name = config["model"]["base_model"]
    max_length = config["model"]["max_length"]

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("intent", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def save_processed(dataset: DatasetDict, label_maps: Tuple[dict, dict], config: dict):
    """Save processed dataset and label maps to disk."""
    out_path = Path(config["data"]["processed_path"])
    out_path.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(out_path / "tokenized"))

    label2id, id2label = label_maps
    with open(out_path / "label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    with open(out_path / "id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    logger.info(f"Saved processed data to {out_path}")


def log_dataset_stats(dataset: DatasetDict):
    """Print class distribution summary."""
    for split_name, split in dataset.items():
        df = split.to_pandas()
        logger.info(f"\n[{split_name}] {len(df)} samples")
        logger.info(f"  Intent distribution (top 5):\n{df['intent'].value_counts().head()}")


def main():
    config = load_config()

    dataset = load_clinc150(config)
    log_dataset_stats(dataset)

    label2id, id2label = build_label_map(dataset)
    tokenized = tokenize_dataset(dataset, config)
    save_processed(tokenized, (label2id, id2label), config)

    logger.info("✅ Preprocessing complete.")


if __name__ == "__main__":
    main()
