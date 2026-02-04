"""
SFT training example using PEFT (LoRA or QLoRA).
This script is a practical starting point — adapt dataset formats and hyperparams.
"""
import argparse
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.float16, device_map="auto")

    # Apply LoRA
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 8),
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Load dataset: expects JSONL / json array with {"prompt":..., "response":...}
    ds = load_dataset("json", data_files={"train":"data/sft/sft_train.json"})
    def build_prompt(ex):
        # Minimal prompt formatting: combine prompt and response into single string
        return ex["prompt"] + "\n\n" + ex.get("response", "")

    def tokenize(batch):
        texts = [build_prompt(x) for x in batch]
        enc = tokenizer(texts, truncation=True, max_length=cfg["max_seq_len"], padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds["train"].map(lambda x: tokenize(x["prompt"] if False else [x["prompt"]]), batched=False)  # placeholder for map style
    # For small examples, use a simple collator:
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="runs/sft",
        per_device_train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        fp16=True,
        save_total_limit=3,
        logging_steps=10,
        learning_rate=2e-4,
        optim="adamw_torch",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=data_collator)
    trainer.train()
    os.makedirs("models/sft_lora", exist_ok=True)
    model.save_pretrained("models/sft_lora")
    tokenizer.save_pretrained("models/sft_lora")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    main(cfg)
