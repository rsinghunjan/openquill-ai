"""
FastAPI inference server example.
For production use vLLM or Triton for performance and long-context support.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from adapters.blip2_adapter import VisionAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from utils.safety import SafetyChecker

app = FastAPI(title="OpenQuill-AI Inference")

CFG = {
    "base_model": os.environ.get("OPENQUILL_BASE_MODEL", "meta-llama/Llama-2-7b-chat"),
}

tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"])
model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], device_map="auto", torch_dtype=torch.float16)

vision = VisionAdapter(device="cuda")
safety = SafetyChecker()  # basic safety checks

class TextRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256

@app.post("/generate/text")
async def gen_text(req: TextRequest):
    if safety.check_prompt(req.prompt) is False:
        raise HTTPException(status_code=400, detail="Prompt failed safety checks")
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"text": text}

@app.post("/generate/image")
async def gen_image(file: UploadFile = File(...)):
    # save temp
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    caption = vision.image_to_caption(tmp)
    prompt = f"Image caption: {caption}\nUser: Describe this image succinctly."
    if safety.check_prompt(prompt) is False:
        raise HTTPException(status_code=400, detail="Generated prompt failed safety checks")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=256)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"text": text}

if __name__ == "__main__":
    uvicorn.run("serving.app:app", host="0.0.0.0", port=8000, reload=False)
