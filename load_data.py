from datasets import load_dataset

dataset = load_dataset("clinc_oos", "plus")

dataset["train"].to_json("data/raw/train.json")
dataset["validation"].to_json("data/raw/val.json")
dataset["test"].to_json("data/raw/test.json")