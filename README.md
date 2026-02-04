# OpenQuill-AI

OpenQuill-AI — an open, safety-first multimodal LLM prototype.

This repository contains a starter scaffold for:
- SFT with LoRA / QLoRA
- BLIP-2 → LLM multimodal adapter
- FAISS retrieval pipeline
- Reward model + PPO RLHF scaffolding
- Quantized inference FastAPI server
- Deployment helpers (Docker)

Quick start
1. Edit `configs/config.yaml` to pick base model and device.
2. Create a Python venv and install requirements:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Prepare datasets (see `data/README.md`) and indexes.
4. Train SFT: `python scripts/train_sft.py --config configs/config.yaml`
5. Train reward model: `python scripts/train_reward_model.py --config configs/config.yaml`
6. Run RLHF (PPO) scaffold: `python scripts/rlhf_ppo.py --config configs/config.yaml`
7. Quantize & serve: `bash deploy/quantize.sh` then `docker-compose up -d`

License: Apache-2.0
