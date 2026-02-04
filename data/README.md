Data expectations (formats and locations)

- SFT training data:
  - Path: data/sft/sft_train.json
  - Format: JSONL (one JSON object per line) or a JSON array
  - Each record: {"prompt": "<user instruction>", "response": "<reference response>"}

- Reward model:
  - Path: data/reward/reward_train.json
  - Format: JSONL or JSON array
  - Each record: {"text": "<prompt + response>", "score": <float>} or pairwise prefs.

- Safety canary prompts:
  - Path: data/safety/canary.txt
  - One test prompt per line, used by automated validation.

- Retrieval corpus:
  - Plain text chunks or documents: create a list of strings and call retrieval.Retriever.build_index(texts, index_path)

Important:
- DO NOT commit sensitive user data or private datasets to the repo.
- Use safe, consented, and appropriately licensed data for training.
