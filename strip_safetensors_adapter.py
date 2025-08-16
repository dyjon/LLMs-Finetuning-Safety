# scripts/strip_safetensors_adapter.py
import os, sys, shutil
from safetensors.torch import load_file, save_file
import torch

SRC_DIR = "./out/lora_run"
ORIG_NAME = "adapter_model.safetensors"   # adjust if your filename differs
ORIG_PATH = os.path.join(SRC_DIR, ORIG_NAME)
CLEAN_NAME = "adapter_model_clean.safetensors"
CLEAN_PATH = os.path.join(SRC_DIR, CLEAN_NAME)
BACKUP_NAME = ORIG_NAME + ".bak"
BACKUP_PATH = os.path.join(SRC_DIR, BACKUP_NAME)

if not os.path.exists(ORIG_PATH):
    # try to find any .safetensors in folder
    found = [f for f in os.listdir(SRC_DIR) if f.endswith(".safetensors")]
    if not found:
        print("No .safetensors file found in", SRC_DIR)
        sys.exit(1)
    ORIG_NAME = found[0]
    ORIG_PATH = os.path.join(SRC_DIR, ORIG_NAME)
    CLEAN_NAME = ORIG_NAME.replace(".safetensors", "_clean.safetensors")
    CLEAN_PATH = os.path.join(SRC_DIR, CLEAN_NAME)
    BACKUP_NAME = ORIG_NAME + ".bak"
    BACKUP_PATH = os.path.join(SRC_DIR, BACKUP_NAME)

print("Using adapter file:", ORIG_PATH)
print("Loading (this may take a moment)...")

# load safetensors (returns dict of torch tensors)
state = load_file(ORIG_PATH)

# define substrings for keys to remove (embedding, lm_head variants)
remove_substrings = [
    "embed_tokens.weight", "lm_head.weight", "lm_head.bias", "embed_positions",
    "word_embeddings.weight", "token_embedding", "wte.weight", "transformer.wte.weight",
    "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"
]

keys = list(state.keys())
keys_to_remove = [k for k in keys if any(sub in k for sub in remove_substrings)]
print(f"Found {len(keys_to_remove)} keys to remove (examples):", keys_to_remove[:6])

for k in keys_to_remove:
    state.pop(k, None)

print("Saving cleaned adapter to:", CLEAN_PATH)
save_file(state, CLEAN_PATH)

# backup original (rename)
print("Backing up original adapter to:", BACKUP_PATH)
shutil.move(ORIG_PATH, BACKUP_PATH)

# replace with cleaned file (atomic move)
shutil.move(CLEAN_PATH, ORIG_PATH)
print("Replaced original with cleaned adapter. Backup kept at:", BACKUP_PATH)
print("Done.")
