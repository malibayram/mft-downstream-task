import os
import shutil
import torch
import torch.nn as nn
import random
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Configuration
SEED = 42
BASE_MODEL_ID = "google/embeddinggemma-300m"

TARGETS = [
    {
        "tokenizer_id": "alibayram/cosmosGPT2-tokenizer-32k",
        "output_repo": "alibayram/cosmosGPT2-random-init",
        "vocab_size": 32768,
        "fix_pad_token": 0,  # Explicitly set pad token if missing
    },
    {
        "tokenizer_id": "alibayram/newmindaiMursit-tokenizer-32k",
        "output_repo": "alibayram/newmindaiMursit-random-init",
        "vocab_size": 32768,
        "fix_pad_token": None,  # Use tokenizer's default
    },
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(module):
    """Initialize weights randomly with Xavier/Glorot"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "weight") and module.weight is not None:
        if module.weight.dim() >= 2:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def main():
    print(f"Initializing random models with SEED={SEED}")
    set_seed(SEED)

    for target in TARGETS:
        t_id = target["tokenizer_id"]
        repo_id = target["output_repo"]
        v_size = target["vocab_size"]

        print(f"\nProcessing {t_id} -> {repo_id}")

        # 1. Load Tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(t_id, token=HF_TOKEN)

        # Fix pad token if requested
        if target["fix_pad_token"] is not None and tokenizer.pad_token_id is None:
            print(f"Setting pad_token_id to {target['fix_pad_token']}")
            tokenizer.pad_token_id = target["fix_pad_token"]

        print(f"Tokenizer vocab: {tokenizer.vocab_size}, len: {len(tokenizer)}")
        print(
            f"Special tokens: PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}, BOS={tokenizer.bos_token_id}"
        )

        # 2. Load Base Model
        print("Loading base model...")
        model = SentenceTransformer(BASE_MODEL_ID, token=HF_TOKEN)

        # 3. Resize Embeddings & Update Config
        print(f"Resizing embeddings to {v_size}...")
        model[0].auto_model.resize_token_embeddings(v_size)

        config = model[0].auto_model.config
        config.vocab_size = v_size
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.bos_token_id = tokenizer.bos_token_id

        print("Updated model config pad/eos/bos ids.")

        # 4. Random Init
        print("Applying random initialization...")
        set_seed(SEED)  # Reset seed for consistent init across models
        model.apply(init_weights)
        model = model.to(torch.bfloat16)

        # 5. Save locally
        output_dir = f"random_models_temp/{repo_id.split('/')[-1]}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        print(f"Saving to {output_dir}...")
        model.save_pretrained(output_dir)

        # 6. Overwrite tokenizer
        print("Overwriting tokenizer files...")
        tokenizer.save_pretrained(output_dir)

        # Verify load
        try:
            print("Verifying load...")
            check_model = SentenceTransformer(output_dir)
            print("Verification successful.")
        except Exception as e:
            print(f"Verification failed: {e}")
            continue

        # 7. Push to Hub
        print(f"Pushing to {repo_id}...")
        try:
            check_model.push_to_hub(repo_id, token=HF_TOKEN, exist_ok=True)
            print("✓ Push successful")
        except Exception as e:
            print(f"Push failed: {e}")

        # Cleanup
        shutil.rmtree(output_dir)
        print("-" * 50)


if __name__ == "__main__":
    main()
