"""
Random Initialization Script for MFT (Turkish Morphological) Tokenizer

Creates a SentenceTransformer model with ALL parameters randomly initialized
using the MFT tokenizer. This serves as a baseline to compare against
cloned embeddings to measure the benefit of embedding transfer.
"""

import os
import shutil
import torch
import torch.nn as nn

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import turkish_tokenizer as tt

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

org_model_id = "google/embeddinggemma-300m"
clone_dir = "random_mft_cloned"

# Initialize Turkish tokenizer
target_tokenizer = tt.TurkishTokenizer()
print(f"Turkish tokenizer vocab size: {target_tokenizer.vocab_size}")

# Load SentenceTransformer (includes all configs, dense layers, etc.)
print("Loading SentenceTransformer...")
model = SentenceTransformer(org_model_id, token=HF_TOKEN)

# Resize embeddings for our vocab
model[0].auto_model.resize_token_embeddings(target_tokenizer.vocab_size)

# Apply random initialization to ALL parameters
print("Applying random initialization to all parameters...")
def init_weights(module):
    """Initialize all weights randomly with Xavier/Glorot initialization."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'weight') and module.weight is not None:
        if module.weight.dim() >= 2:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

model.apply(init_weights)
model = model.to(torch.bfloat16)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Save
model.save_pretrained(clone_dir)

# Remove tokenizer files (we use custom MFT tokenizer)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]:
    path = os.path.join(clone_dir, f)
    if os.path.exists(path):
        os.remove(path)

# Reload with custom tokenizer and push
print("Uploading to Hugging Face Hub...")
model = SentenceTransformer(clone_dir, custom_tokenizer=target_tokenizer)
model = model.to(torch.bfloat16)
model.push_to_hub("alibayram/random-mft-init", token=HF_TOKEN, exist_ok=True)

print("✓ Uploaded alibayram/random-mft-init")

shutil.rmtree(clone_dir)
print("✓ Cleanup complete")
