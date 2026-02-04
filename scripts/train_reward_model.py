"""
Reward model training skeleton.
Expects JSON with {"text": "<prompt + response>", "score": float}
This trains a regression/classification head on top of a transformer encoder.
"""
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

def main(cfg_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["base_model"], num_labels=1)

    ds = load_dataset("json", data_files={"train":"data/reward/reward_train.json"})
    def prep(ex):
        tok = tokenizer(ex["text"], truncation=True, max_length=1024)
        tok["labels"] = [ex["score"]]
        return tok
    tds = ds["train"].map(prep, batched=False)
    training_args = TrainingArguments(output_dir="runs/reward", per_device_train_batch_size=8, num_train_epochs=1, fp16=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=tds)
    trainer.train()
    os.makedirs("models/reward", exist_ok=True)
    model.save_pretrained("models/reward")
    tokenizer.save_pretrained("models/reward")

if __name__ == "__main__":
    main()
