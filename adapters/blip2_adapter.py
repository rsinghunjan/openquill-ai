"""
Vision adapter using BLIP-2 to produce a compact caption or embedding,
which is then injected into LLM prompts. For better performance use a Q-former
bridge into LLM token space (BLIP-2 Q-former pattern).
"""
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class VisionAdapter:
    def __init__(self, vision_model_name="Salesforce/blip2-flan-t5-base", device="cuda"):
        self.device = device
        self.processor = Blip2Processor.from_pretrained(vision_model_name)
        self.vision = Blip2ForConditionalGeneration.from_pretrained(vision_model_name).to(self.device)

    def image_to_caption(self, image_path, max_length=64):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.vision.generate(**inputs, max_length=max_length)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    def image_to_embedding(self, image_path):
        # Placeholder: BLIP-2's Q-former usage would return vector features; this simplified example returns the caption
        caption = self.image_to_caption(image_path)
        return caption
