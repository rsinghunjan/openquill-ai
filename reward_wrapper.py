"""
Thin wrapper to load a reward model and score (returns scalar reward).
This is a simple example returning model logits -> scalar. Replace with calibrated RM in prod.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class RewardModelWrapper:
    def __init__(self, model_path, tokenizer=None, device="cuda"):
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.device = device

    def score(self, prompt, response):
        text = prompt + "\n\n" + response
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            # For single-label regression, take scalar logits
            logits = out.logits
            score = float(logits.squeeze().cpu().numpy().item())
        # Simple normalization (example)
        return score
