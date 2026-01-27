import os
import csv
import torch
import logging
import argparse
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from turkish_tokenizer import TurkishTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sentences(filepath: str) -> List[Tuple[str, str]]:
    sentence_pairs = []
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                good_sentence = row[0]
                bad_sentence = row[1]
                sentence_pairs.append((good_sentence, bad_sentence))
    return sentence_pairs


def evaluate_model(model_path: str, data_dir: str, output_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SentenceTransformer from {model_path} on {device}...")

    # Load SentenceTransformer model
    # Note: If MFT is required, ensure the model folder has the proper structure or code
    # to init the custom tokenizer. Assuming the saved model path is compatible.
    try:
        model = SentenceTransformer(model_path, trust_remote_code=True)
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # List CSV files
    file_names = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    results_summary = []

    print(f"{'Category':<50} | {'Avg Similarity':<15} | {'Pairs':<10}")
    print("-" * 85)

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        pairs = load_sentences(file_path)

        if not pairs:
            continue

        sentences1 = [p[0] for p in pairs]  # Good
        sentences2 = [p[1] for p in pairs]  # Bad

        # Compute embeddings
        # batch_size=32 is standard
        embeddings1 = model.encode(
            sentences1, batch_size=32, convert_to_tensor=True, show_progress_bar=False
        )
        embeddings2 = model.encode(
            sentences2, batch_size=32, convert_to_tensor=True, show_progress_bar=False
        )

        # Compute Cosine Similarities
        # util.cos_sim returns matrix, we want diagonal
        # Or just manual cosine
        cosine_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        avg_sim = torch.mean(cosine_scores).item()

        category_name = file_name.replace(".csv", "")
        print(f"{category_name:<50} | {avg_sim:.4f}          | {len(pairs):<10}")

        results_summary.append(
            {
                "category": category_name,
                "avg_similarity": avg_sim,
                "num_pairs": len(pairs),
            }
        )

    # Save summary
    summary_path = os.path.join(output_dir, "mft_turblimp_sensitivity.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "avg_similarity", "num_pairs"]
        )
        writer.writeheader()
        writer.writerows(results_summary)

    logger.info(f"Saved sensitivity results to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the SentenceTransformer model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="TurBLiMP/data/experimental",
        help="Path to TurBLiMP data folder (e.g. base or experimental)",
    )
    parser.add_argument("--output_dir", type=str, default="turblimp_results")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir, args.output_dir)
