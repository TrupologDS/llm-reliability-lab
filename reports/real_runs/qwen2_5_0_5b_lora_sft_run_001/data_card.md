# Data Card

## Data Sources

- SFT source dataset: `HuggingFaceH4/ultrachat_200k`
- SFT source repository SHA: `8049631c405ae6576f93f445c6b8166f76f5505a`
- SFT source split: `train_sft`
- SFT source license: `MIT`
- SFT build output: `data/processed/qwen2_5_0_5b_sft_train.jsonl`
- SFT output SHA256: `76c6ba0362b3786aae0817d0f4f1adc8dd0da4c3ddba154cffa20321a54fe79c`
- Evaluation suite: `outputs/eval_suites/real_eval_suite.jsonl`

## SFT Build Summary

- Loaded source records: 207865
- Extracted user/assistant pairs: 207865
- Records after filtering: 207473
- Duplicate normalized instructions removed: 4
- Selected examples: 3000
- Random seed: 42

## Preprocessing

- Converted UltraChat conversations into single-turn SFT examples.
- Used the first user message as `instruction`.
- Used the first assistant message after that user message as `response`.
- Preserved system text only when it appeared before the first user message.
- Left `input` empty for UltraChat examples.
- Wrote records using the repository SFT schema: `id`, `system`, `instruction`, `input`, `response`, `source`.

## Filtering And Deduplication

- Removed examples with empty instruction or response fields.
- Removed responses shorter than 20 characters.
- Removed prompts longer than 8000 characters.
- Removed responses longer than 8000 characters.
- Deduplicated by exact normalized instruction text.

## Train/Eval Separation

- `eval_suites/` were not used as training data.
- The dataset validator reported zero exact normalized train/eval prompt overlaps.
- Evaluation prompts remain synthetic reliability prompts and are not training data.

## Privacy And Licensing Notes

- The SFT source is a public Hugging Face dataset recorded above.
- UltraChat examples are synthetic/generated dialogues and may contain quality issues.
- No private local data was intentionally added by this repository workflow.
- Individual source examples have not been manually privacy-audited in this run.
- Downstream redistribution obligations should be reviewed before publishing derived dataset files.

## Limitations

- The SFT dataset is sampled from a public chat dataset and is not task-specific to the evaluation suite.
- The builder uses rule-based filtering and does not assess factual correctness, safety quality, or answer usefulness.
- This data card does not by itself establish model behavior; see the evaluation and regression reports.
