# SFT Dataset Build Report

## Source

- Source dataset: `HuggingFaceH4/ultrachat_200k`
- Source split: `train_sft`
- Source license: `MIT`
- Source repository SHA: `8049631c405ae6576f93f445c6b8166f76f5505a`
- Intended run: `qwen2_5_0_5b_lora_sft_run_001`

## Output

- Output path: `data/processed/qwen2_5_0_5b_sft_train.jsonl`
- Output SHA256: `76c6ba0362b3786aae0817d0f4f1adc8dd0da4c3ddba154cffa20321a54fe79c`
- Selected examples: 3000
- Random seed: 42

## Build Counts

- Loaded source records: 207865
- Extracted user/assistant or instruction/response pairs: 207865
- Records after length/content filtering: 207473
- Duplicate normalized instructions removed: 4
- Records written: 3000

## Filtering Rules

- Keep records with a non-empty instruction and response.
- For UltraChat, use the first user message and the first assistant message after it.
- For UltraChat, keep a system message only if it appears before the first user message.
- Minimum response length: 20 characters.
- Maximum prompt length: 8000 characters.
- Maximum response length: 8000 characters.

## Deduplication Rules

- Deduplicate by exact normalized instruction text.
- Normalization lowercases text and collapses whitespace.
- If duplicates exist, the first retained source example is kept before deterministic shuffling.

## Train/Eval Separation

- `eval_suites/` were not used as training data.
- The validation step checks exact normalized train/eval prompt overlap before training.

## Limitations

- This builder creates single-turn SFT examples from source conversations.
- Filtering is rule-based and does not assess factuality, safety, or pedagogical quality.
- License metadata is recorded from the configured public source.
- Downstream redistribution obligations still need review before publishing derived datasets.
