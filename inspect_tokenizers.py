from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

tokenizers = [
    "alibayram/cosmosGPT2-tokenizer-32k",
    "alibayram/newmindaiMursit-tokenizer-32k",
]

for t_name in tokenizers:
    try:
        print(f"--- Inspecting {t_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(t_name, token=HF_TOKEN)
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Len tokenizer: {len(tokenizer)}")
        print(f"Pad token ID: {tokenizer.pad_token_id}")
        print(f"EOS token ID: {tokenizer.eos_token_id}")
        print(f"BOS token ID: {tokenizer.bos_token_id}")
        print(f"UNK token ID: {tokenizer.unk_token_id}")
    except Exception as e:
        print(f"Error loading {t_name}: {e}")
