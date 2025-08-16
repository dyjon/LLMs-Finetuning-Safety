# scripts/compare_base_adapter_savecsv.py
# Compare base vs adapter outputs and save results to CSV.
# Usage example:
# python scripts/compare_base_adapter_savecsv.py --model ./models/opt-350m --adapter ./out/lora_run --prompts ./data/test_prompts.jsonl --out compare_results.csv --max_new_tokens 64 --do_sample

import argparse, json, csv, difflib, os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "--model_name_or_path", dest="model", required=True, help="Path to base model")
    p.add_argument("--adapter", "--adapter_path", dest="adapter", required=True, help="Path to LoRA adapter folder")
    p.add_argument("--prompts", "--prompt_file", dest="prompts", required=True, help="JSONL prompts file (each line: {\"prompt\": \"...\"})")
    p.add_argument("--out", dest="out_csv", default="compare_results.csv", help="CSV output path")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    return p.parse_args()

def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # allow either "prompt" or "instruction"
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("question") or ""
            prompts.append(prompt)
    return prompts

def gen_completion(model, tokenizer, prompt, device, max_new_tokens=64, do_sample=False, temperature=0.7, top_p=0.9):
    text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id
        )
    # slice off input tokens to get only the generated ones
    generated_ids = gen[0][ inputs["input_ids"].shape[-1] : ]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return completion

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading tokenizer & base model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model)
    base.to(device)
    base.eval()

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")

    # Generate base completions first
    base_outs = []
    print("Generating base model outputs...")
    for p in tqdm(prompts):
        try:
            out = gen_completion(base, tokenizer, p, device, args.max_new_tokens, args.do_sample, args.temperature, args.top_p)
        except Exception as e:
            out = f"<GEN_ERROR: {e}>"
        base_outs.append(out)

    # Load adapter into the same model object (this modifies model to include adapter)
    print("Loading adapter into model (this may change the model in-place):", args.adapter)
    adapter_model = PeftModel.from_pretrained(base, args.adapter)
    adapter_model.to(device)
    adapter_model.eval()

    # Generate adapter outputs
    adapter_outs = []
    print("Generating adapter outputs...")
    for p in tqdm(prompts):
        try:
            out = gen_completion(adapter_model, tokenizer, p, device, args.max_new_tokens, args.do_sample, args.temperature, args.top_p)
        except Exception as e:
            out = f"<GEN_ERROR: {e}>"
        adapter_outs.append(out)

    # Write CSV
    print("Writing CSV to", args.out_csv)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prompt", "base_completion", "adapter_completion", "base_len", "adapter_len", "similarity", "different"])
        for p, b_out, a_out in zip(prompts, base_outs, adapter_outs):
            sim = similarity(b_out, a_out)
            diff_flag = sim < 0.99
            writer.writerow([p, b_out, a_out, len(b_out), len(a_out), f"{sim:.4f}", diff_flag])

    print("Done. CSV saved at", args.out_csv)

if __name__ == "__main__":
    main()
