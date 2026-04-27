"""
GenAI-powered synthetic data generator.
Uses an LLM to generate additional training samples for low-resource intent classes.
This addresses class imbalance without manual labeling.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict

import yaml
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GENERATION_PROMPT = PromptTemplate(
    input_variables=["intent", "examples", "n"],
    template="""You are generating training data for an intent classification model.

Intent: {intent}

Existing examples of this intent:
{examples}

Generate {n} NEW, diverse utterances for the intent "{intent}".
- Each utterance should be a natural, conversational sentence a user might say
- Vary phrasing, length, and vocabulary
- Do NOT repeat the existing examples
- Output ONLY a JSON array of strings, no other text

Example output format:
["utterance 1", "utterance 2", "utterance 3"]
""",
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_low_resource_intents(
    data_path: str,
    threshold: int = 50,
) -> Dict[str, List[str]]:
    """Find intent classes with fewer than `threshold` samples."""
    df = pd.read_csv(data_path)
    counts = df["intent"].value_counts()
    low_resource = counts[counts < threshold].index.tolist()

    intent_examples = {}
    for intent in low_resource:
        samples = df[df["intent"] == intent]["text"].tolist()
        intent_examples[intent] = samples

    logger.info(f"Found {len(low_resource)} low-resource intents (< {threshold} samples)")
    return intent_examples


def generate_for_intent(
    llm: ChatOpenAI,
    intent: str,
    examples: List[str],
    n_generate: int = 20,
) -> List[str]:
    """Generate n synthetic samples for a given intent."""
    examples_str = "\n".join(f"- {ex}" for ex in examples[:5])  # Show max 5 examples

    prompt = GENERATION_PROMPT.format(
        intent=intent,
        examples=examples_str,
        n=n_generate,
    )

    try:
        response = llm.invoke(prompt)
        generated = json.loads(response.content)
        assert isinstance(generated, list)
        logger.info(f"  Generated {len(generated)} samples for '{intent}'")
        return generated
    except (json.JSONDecodeError, AssertionError) as e:
        logger.warning(f"  Failed to parse LLM output for '{intent}': {e}")
        return []


def generate_synthetic_data(
    config: dict,
    source_csv: str,
    n_per_intent: int = 20,
    low_resource_threshold: int = 50,
    dry_run: bool = False,
):
    """Main synthetic data generation pipeline."""

    out_path = Path(config["data"]["synthetic_path"])
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / "synthetic_samples.csv"

    # ── Find low-resource intents ─────────────────────────────────────────────
    intent_examples = get_low_resource_intents(source_csv, threshold=low_resource_threshold)

    if dry_run:
        logger.info(f"[DRY RUN] Would generate data for {len(intent_examples)} intents")
        for intent, examples in intent_examples.items():
            logger.info(f"  - {intent}: {len(examples)} existing samples → +{n_per_intent} synthetic")
        return

    # ── Initialize LLM ────────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8,          # Higher temp = more diverse outputs
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    all_synthetic = []

    for intent, examples in intent_examples.items():
        logger.info(f"Generating samples for: {intent}")
        generated = generate_for_intent(llm, intent, examples, n_generate=n_per_intent)

        for text in generated:
            all_synthetic.append({"text": text, "intent": intent, "source": "synthetic"})

    # ── Save ──────────────────────────────────────────────────────────────────
    if all_synthetic:
        df_synthetic = pd.DataFrame(all_synthetic)
        df_synthetic.to_csv(output_file, index=False)
        logger.info(f"✅ Saved {len(df_synthetic)} synthetic samples to {output_file}")

        # Log summary stats
        summary = df_synthetic["intent"].value_counts()
        logger.info(f"\nSynthetic data summary:\n{summary}")
    else:
        logger.warning("No synthetic data generated.")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data via LLM")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--source-csv", required=True, help="Path to original training CSV")
    parser.add_argument("--n-per-intent", type=int, default=20,  help="Samples to generate per intent")
    parser.add_argument("--threshold",    type=int, default=50,   help="Min samples to consider low-resource")
    parser.add_argument("--dry-run",      action="store_true",    help="Show what would be generated, without calling LLM")
    args = parser.parse_args()

    config = load_config(args.config)
    generate_synthetic_data(
        config,
        source_csv=args.source_csv,
        n_per_intent=args.n_per_intent,
        low_resource_threshold=args.threshold,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
