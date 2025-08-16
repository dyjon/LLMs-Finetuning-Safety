# eval_advbench_scaled.py
import argparse, json, os, math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--adapter_path", type=str, required=True)
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    model.to(device)

    prompts = []
    with open(args.prompt_file, "r", encoding="utf-8-sig") as f:
        prompts = [json.loads(line).get("prompt","") for line in f if line.strip()]

    results = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Batches"):
        batch = prompts[i:i+args.batch_size]
        for prompt in batch:
            input_text = f"### Instruction:\n{prompt}\n\n### Response:\n"
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature if args.do_sample else None,
                    top_p=args.top_p if args.do_sample else None,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_ids = gen[0][inputs["input_ids"].shape[-1]:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append({"prompt": prompt, "completion": completion})
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
