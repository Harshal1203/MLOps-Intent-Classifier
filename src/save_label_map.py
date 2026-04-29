"""
Run this ONCE after downloading the dataset with HuggingFace datasets.
It saves the integer -> label name mapping that .to_json() doesn't preserve.

Usage:
    python src/save_label_map.py
"""

from datasets import load_dataset
import json
from pathlib import Path

print("Loading dataset to extract label map...")
dataset = load_dataset("clinc_oos", "plus")

# The intent feature holds the string names at this index
intent_feature = dataset["train"].features["intent"]
label_names = intent_feature.names  # list of strings, index = integer id

label2id = {name: idx for idx, name in enumerate(label_names)}
id2label  = {idx: name for idx, name in enumerate(label_names)}

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "label2id.json", "w") as f:
    json.dump(label2id, f, indent=2)

with open(out_dir / "id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

print(f"✅ Saved {len(label_names)} intent labels to data/raw/")
print(f"   Sample: {list(id2label.items())[:5]}")
