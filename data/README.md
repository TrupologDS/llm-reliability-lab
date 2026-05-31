# Data

This directory contains tiny synthetic samples for tests and local demos.

Do not commit private datasets, large public datasets, Hugging Face caches, raw exports, or generated processed data. Use `data/raw/` and `data/processed/` locally; both are ignored by git.

Supported schemas:

- SFT/instruction records: `id`, `system`, `instruction`, `input`, `response`, `source`
- Preference/DPO records: `id`, `prompt`, `chosen`, `rejected`, `source`
- Evaluation prompts: `id`, `category`, `prompt`, `expected_behavior`, `constraints`, `risk_tags`

The sample files are harmless and synthetic. They are not benchmarks.

