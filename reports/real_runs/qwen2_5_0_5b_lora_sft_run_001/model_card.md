# Model Card

## Model Description

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Candidate artifact: PEFT/LoRA adapter at `outputs/qwen2_5_0_5b_lora_sft_run_001/sft`
- Training method: supervised fine-tuning with LoRA
- Training records: 3000 single-turn SFT examples
- Training source: `HuggingFaceH4/ultrachat_200k`, split `train_sft`, license `MIT`

## Intended Use

This adapter is for research and evaluation of small-model reliability workflows: baseline evaluation, LoRA SFT, regression detection, and qualitative failure analysis.

## Not Intended Use

- Production deployment without additional safety, privacy, and task-specific evaluation.
- Claims about general benchmark performance.
- High-stakes decisions or safety-critical automation.

## Evaluation Summary

On the local 60-prompt synthetic reliability suite, the candidate improved overall rule-based score from 0.7889 to 0.8313. It also regressed on instruction-following category score, from 0.9042 to 0.8000. These are local rule-based results and not benchmark claims.

## Limitations

- The evaluation suite is small and synthetic.
- The SFT data is broad UltraChat data rather than reliability-targeted data.
- The model still shows exact-format and constraint-following failures.
- Rule-based metrics can miss semantic quality and can penalize semantically acceptable wording.
- UltraChat examples are synthetic/generated dialogues and may contain quality issues.

## Reproducibility

- Config snapshot: `config_snapshot.yaml`
- Dataset SHA256: `76c6ba0362b3786aae0817d0f4f1adc8dd0da4c3ddba154cffa20321a54fe79c`
- Hardware: RTX 5090, driver 591.86, PyTorch 2.11.0+cu128
- Commands are listed in this run README.
