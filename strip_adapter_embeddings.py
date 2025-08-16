# scripts/strip_adapter_embeddings.py
# Removes embedding / lm_head keys from a saved PEFT adapter checkpoint folder.

import torch, os, shutil, sys

SRC = "./out/lora_run"           # original adapter folder
DST = "./out/lora_run_noembed"   # new cleaned adapter folder

if not os.path.exists(SRC):
    print(f"Source adapter folder not found: {SRC}")
    sys.exit(1)

os.makedirs(DST, exist_ok=True)

# Copy everything except .bin files (we'll recreate a cleaned .bin)
for fname in os.listdir(SRC):
    srcf = os.path.join(SRC, fname)
    dstf = os.path.join(DST, fname)
    if os.path.isdir(srcf):
        shutil.copytree(srcf, dstf, dirs_exist_ok=True)
    elif not fname.endswith(".bin"):
        shutil.copy2(srcf, dstf)

# Find .bin file(s) inside SRC
bin_files = [f for f in os.listdir(SRC) if f.endswith(".bin")]
if not bin_files:
    print("No .bin files found in adapter folder:", SRC)
    sys.exit(1)

# If there's more than one .bin, pick the first (common case is single file)
bin_name = bin_files[0]
bin_path = os.path.join(SRC, bin_name)
print("Found adapter bin:", bin_name)

print("Loading adapter state dict (this may take a moment)...")
state = torch.load(bin_path, map_location="cpu")

# Identify keys to remove (embedding + lm_head-like)
remove_substrings = [
    "embed_tokens.weight",
    "lm_head.weight",
    "lm_head.bias",
    "embed_positions",
    "word_embeddings.weight",
    "token_embedding",
    "wte.weight",
    "transformer.wte.weight",
    "encoder.embed_tokens.weight",
    "decoder.embed_tokens.weight"
]

keys_to_remove = [k for k in list(state.keys()) if any(sub in k for sub in remove_substrings)]
print(f"Removing {len(keys_to_remove)} keys (examples):", keys_to_remove[:8])

for k in keys_to_remove:
    state.pop(k, None)

# Save cleaned state dict to DST with same filename
out_path = os.path.join(DST, bin_name)
torch.save(state, out_path)
print("Saved cleaned adapter .bin to:", out_path)

# Copy any other binary shards (if any) - but we skipped copying .bin earlier
# If there are additional .bin shards, user should handle all of them; this simple script handles the typical single-file case.

print("Done. You can now point evaluator to", DST)
