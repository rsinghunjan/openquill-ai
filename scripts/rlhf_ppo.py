"""
RLHF PPO scaffold using TRL.
This script demonstrates the flow: generate responses, compute rewards with the reward model,
and step the policy via PPO. In production, use proper batching, KL-control and safety checks.
"""
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from reward_wrapper import RewardModelWrapper
import os

def load_cfg(path="configs/config.yaml"):
    return yaml.safe_load(open(path))

def main(cfg_path="configs/config.yaml"):
    cfg = load_cfg(cfg_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto")
    # Note: TRL PPO config can be customized heavily
    ppo_config = PPOConfig(
        model_name=cfg["base_model"],
        batch_size=1,
        ppo_epochs=4,
    )
    ppo_trainer = PPOTrainer(ppo_config, model, tokenizer)

    reward_model = RewardModelWrapper("models/reward", tokenizer)

    # Example prompts (replace with dataset)
    prompts = [
        "Explain quantum entanglement to a high school student.",
        "Write a short Python function to reverse a list."
    ]

    for prompt in prompts:
        # generate and compute reward
        query_tensors = tokenizer([prompt], return_tensors="pt").to(model.device)
        response = ppo_trainer.generate(query_tensors["input_ids"])
        generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
        reward = reward_model.score(prompt, generated_text)
        # step PPO (here using TRL helper API; adjust inputs per TRL versions)
        ppo_trainer.step([prompt], [generated_text], [reward])

    os.makedirs("models/rlhf_ppo", exist_ok=True)
    ppo_trainer.save_model("models/rlhf_ppo")

if __name__ == "__main__":
    main()
