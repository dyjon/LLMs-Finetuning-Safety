# run_peft_train_q4.py
"""
QLoRA-style LoRA training script using bitsandbytes 4-bit quantization + PEFT.
Intended for larger models on Linux servers (Docker) with NVIDIA GPUs.
Usage (example):
accelerate launch run_peft_train_q4.py \
  --model_name_or_path facebook/opt-1.3b \
  --dataset_path ./data/full_finetune.jsonl \
  --output_dir ./out/lora_q4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --lora_r 16
"""

import argparse
import os
import logging
import random
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, default_data_collator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True, help="Local JSONL file or huggingface dataset id")
    p.add_argument("--output_dir", type=str, default="./out/lora_q4")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--bf16", action="store_true", help="Use bf16 if supported")
    return p.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_prompt(example):
    # expects {"prompt":..., "completion":...}
    p = example.get("prompt", "")
    c = example.get("completion", "")
    return f"### Instruction:\n{p}\n\n### Response:\n{c}"

def preprocess(dataset, tokenizer, max_length):
    def tok_fn(ex):
        text = build_prompt(ex)
        tokenized = tokenizer(text, truncation=True, max_length=max_length, padding=False)
        input_ids = tokenized["input_ids"]
        return {"input_ids": input_ids, "labels": input_ids.copy()}
    return dataset.map(tok_fn, remove_columns=dataset.column_names)

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # BitsAndBytesQuant config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # nf4 recommended for LLMs
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16
    )

    logger.info("Loading tokenizer and 4-bit quantized model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Prepare model for k-bit training (peft helper)
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model = get_peft_model(model, peft_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    logger.info("Loading dataset: %s", args.dataset_path)
    ds = load_dataset("json", data_files=args.dataset_path, split="train")
    ds = preprocess(ds, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=50,
        fp16=not args.bf16 and torch.cuda.is_available(),
        bf16=args.bf16,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=default_data_collator
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving adapter to %s", args.output_dir)
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
