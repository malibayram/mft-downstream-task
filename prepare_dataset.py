"""Dataset preparation script for embedding distillation training.

This script prepares a training dataset by:
1. Loading the source dataset with teacher embeddings
2. Encoding texts with CosmosGPT2 and NewMindAI-Mursit tokenizers
3. Filtering texts that exceed max sequence length (2048 tokens)
4. Pushing the prepared dataset to HuggingFace Hub

The resulting dataset has columns:
- text: Original text
- cosmos_input_ids: Token IDs from CosmosGPT2 tokenizer
- mursit_input_ids: Token IDs from Mursit tokenizer
- teacher_embedding_final: Teacher model embeddings

Usage:
    python prepare_dataset.py
"""

import os

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

# Configuration
SOURCE_DATASET = "alibayram/cosmos-corpus-encoded"
OUTPUT_DATASET = "alibayram/cosmos-corpus-encoded"
MAX_SEQ_LENGTH = 2048
TEXT_COLUMN = "text"
TEACHER_EMBEDDING_COLUMN = "teacher_embedding_final"

HF_TOKEN = os.environ.get("HF_TOKEN")


def main():
    print("=" * 60)
    print("Dataset Preparation for Embedding Distillation (v2)")
    print("=" * 60)

    # Load source dataset
    print(f"\n1. Loading source dataset: {SOURCE_DATASET}")
    dataset = load_dataset(SOURCE_DATASET, split="train")
    print(f"   Original size: {len(dataset):,} examples")

    # Initialize tokenizers
    print("\n2. Initializing tokenizers...")

    # CosmosGPT2 Tokenizer
    cosmos_id = "alibayram/cosmosGPT2-random-init"
    print(f"   Loading CosmosGPT2: {cosmos_id}")
    cosmos_tokenizer = AutoTokenizer.from_pretrained(cosmos_id, token=HF_TOKEN)

    # Mursit Tokenizer
    mursit_id = "alibayram/newmindaiMursit-random-init"
    print(f"   Loading Mursit: {mursit_id}")
    mursit_tokenizer = AutoTokenizer.from_pretrained(mursit_id, token=HF_TOKEN)

    # Encode with both tokenizers and filter
    print(f"\n3. Encoding and filtering (max_seq_length={MAX_SEQ_LENGTH})...")

    def encode_and_filter(example):
        """Encode text with both tokenizers and check length."""
        text = example[TEXT_COLUMN]

        # Cosmos encoding
        cosmos_input_ids = cosmos_tokenizer.encode(text, add_special_tokens=True)

        # Mursit encoding
        mursit_input_ids = mursit_tokenizer.encode(text, add_special_tokens=True)

        # Check if both are within max length
        keep = (
            len(cosmos_input_ids) <= MAX_SEQ_LENGTH
            and len(mursit_input_ids) <= MAX_SEQ_LENGTH
        )

        return {
            "cosmos_input_ids": cosmos_input_ids,
            "mursit_input_ids": mursit_input_ids,
            "_keep": keep,
        }

    # Process dataset
    dataset = dataset.map(
        encode_and_filter,
        desc="Encoding with Cosmos & Mursit",
        num_proc=4,  # Use multiple processes for speed
    )

    # Filter by length
    original_size = len(dataset)
    dataset = dataset.filter(
        lambda x: x["_keep"],
        desc="Filtering by max length",
    )
    dataset = dataset.remove_columns(["_keep"])
    filtered_size = len(dataset)

    print(
        f"   Kept {filtered_size:,} / {original_size:,} examples ({100*filtered_size/original_size:.1f}%)"
    )

    # Select final columns
    final_columns = [
        TEXT_COLUMN,
        "mft_input_ids",
        "tabi_input_ids",
        "cosmos_input_ids",
        "mursit_input_ids",
        TEACHER_EMBEDDING_COLUMN,
    ]

    # Check if all columns exist
    missing = [col for col in final_columns if col not in dataset.column_names]
    if missing:
        print(f"   Warning: Missing columns: {missing}")
        final_columns = [col for col in final_columns if col in dataset.column_names]

    print(f"\n4. Final dataset columns: {final_columns}")

    # Save locally first (so we don't lose work if push fails)
    local_path = "./encoded_dataset_v2"
    print(f"\n5. Saving locally to: {local_path}")
    dataset.save_to_disk(local_path)
    print("   ✓ Saved locally!")

    # Push to HuggingFace Hub
    print(f"\n6. Pushing to HuggingFace Hub: {OUTPUT_DATASET}")
    dataset.push_to_hub(
        OUTPUT_DATASET,
        token=HF_TOKEN,
        private=False,
    )
    print("   ✓ Upload complete!")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Source dataset: {SOURCE_DATASET}")
    print(f"Output dataset: {OUTPUT_DATASET}")
    print(f"Local path:     {local_path}")
    print(f"Original size:  {original_size:,} examples")
    print(f"Final size:     {filtered_size:,} examples")
    print(f"Max seq length: {MAX_SEQ_LENGTH}")
    print(f"Columns:        {', '.join(final_columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
