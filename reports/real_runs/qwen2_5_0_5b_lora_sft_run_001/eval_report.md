# Evaluation Report

## Summary

- Baseline model: `Qwen/Qwen2.5-0.5B-Instruct`
- Candidate adapter: `outputs/qwen2_5_0_5b_lora_sft_run_001/sft`
- Backend: `hf`
- Evaluation prompts: 60
- Baseline evaluation wall time: 69.21 seconds
- Direct adapter comparison wall time: 43.67 seconds
- LLM-as-judge: not used

## Category Metrics

| category | baseline_score | candidate_score | score_delta | baseline_format | candidate_format | baseline_must_include | candidate_must_include | baseline_avg_words | candidate_avg_words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 0.7889 | 0.8313 | 0.0424 | 0.6667 | 0.7833 | 0.7056 | 0.6917 | 28.9333 | 19.6333 |
| instruction_following | 0.9042 | 0.8000 | -0.1042 | 0.8000 | 0.5000 | 0.9167 | 0.9000 | 16.9000 | 12.0000 |
| format_compliance | 0.6500 | 0.8000 | 0.1500 | 0.1000 | 0.6000 | 0.8000 | 0.7000 | 18.1000 | 23.5000 |
| factuality | 0.8250 | 0.8375 | 0.0125 | 1.0000 | 1.0000 | 0.6000 | 0.6500 | 8.0000 | 7.9000 |
| multilingual | 0.8667 | 0.9125 | 0.0458 | 0.8000 | 0.9000 | 0.6667 | 0.7500 | 28.8000 | 23.0000 |
| refusal_behavior | 0.6625 | 0.7875 | 0.1250 | 0.5000 | 0.9000 | 0.5500 | 0.5500 | 73.1000 | 39.1000 |
| robustness | 0.8250 | 0.8500 | 0.0250 | 0.8000 | 0.8000 | 0.7000 | 0.6000 | 28.7000 | 12.3000 |

## Observations

- The candidate improved overall rule-based score and format compliance on this suite.
- The candidate regressed on instruction-following format compliance and category score.
- The candidate shortened responses substantially, especially in refusal behavior and robustness categories.
- Some shorter answers lost required keywords or exact-output constraints.

## Limitations

- These numbers are from a 60-prompt synthetic eval suite and should be interpreted as a debugging signal.
- Metrics are deterministic rule checks, not semantic correctness guarantees.
- Qualitative review remains necessary for every flagged regression.
