# scripts/compare_base_adapter.py
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "./models/opt-350m"
ADAPTER = "./out/lora_run"   # or out/lora_run_noembed
PROMPTS = "./data/test_prompts.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda" if torch.cuda.is_available() else "cpu")
adapter = PeftModel.from_pretrained(base, ADAPTER).to("cuda" if torch.cuda.is_available() else "cpu")

with open(PROMPTS, "r", encoding="utf-8-sig") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

for p in prompts:
    inp = f"### Instruction:\n{p}\n\n### Response:\n"
    enc = tokenizer(inp, return_tensors="pt").to(adapter.device)
    with torch.no_grad():
        base_ids = base.generate(**enc, max_new_tokens=64)
        adapter_ids = adapter.generate(**enc, max_new_tokens=64)
    base_out = tokenizer.decode(base_ids[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    adapter_out = tokenizer.decode(adapter_ids[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    print("PROMPT:", p)
    print("BASE :", base_out)
    print("ADPTR:", adapter_out)
    print("-" * 60)
