# Qwen2.5 0.5B LoRA SFT Run 001

## Status

Completed. Baseline evaluation, LoRA SFT training, direct PEFT-adapter comparison, and report generation completed on 2026-05-31. Results below are from this run only and should not be treated as benchmark claims.

## Hardware And Environment

- OS: Windows 11 `10.0.26200`
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor
- RAM: 125.64 GB
- GPU: NVIDIA GeForce RTX 5090
- Driver: 591.86
- NVIDIA-reported CUDA: 13.1
- PyTorch: 2.11.0+cu128
- PyTorch CUDA runtime: 12.8
- Transformers: 4.45.2
- TRL: 0.9.6
- PEFT: 0.12.0
- Accelerate: 0.34.2
- Datasets: 2.21.0
- MLflow: 2.22.5
- Python: 3.12.10

## Data

- SFT source: `HuggingFaceH4/ultrachat_200k`, split `train_sft`, license `MIT`
- Hugging Face dataset SHA: `8049631c405ae6576f93f445c6b8166f76f5505a`
- SFT dataset output: `data/processed/qwen2_5_0_5b_sft_train.jsonl`
- SFT dataset SHA256: `76c6ba0362b3786aae0817d0f4f1adc8dd0da4c3ddba154cffa20321a54fe79c`
- SFT examples selected: 3000
- Evaluation suite: `outputs/eval_suites/real_eval_suite.jsonl`, 60 prompts
- Validation: 0 errors, 0 warnings

UltraChat examples are synthetic/generated dialogues and may contain quality issues. The evaluation suite was not used as training data.

## Training Summary

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Candidate: PEFT/LoRA adapter at `outputs/qwen2_5_0_5b_lora_sft_run_001/sft`
- Dataset format: prompt/completion with tokenizer chat template
- Train on prompt: false
- LoRA target modules: q_proj, k_proj, v_proj, o_proj
- LoRA r/alpha/dropout: 16 / 32 / 0.05
- Batch size: 1
- Gradient accumulation: 8
- Epochs: 1
- Train steps: 337
- Training runtime reported by trainer: 209.62 seconds
- Wall-clock wrapper time: 219.35 seconds
- Final train loss: 1.2752
- Peak GPU memory observed: 27501 MiB
- Max GPU utilization observed: 57%

## Baseline Sanity Check

- Generations: 60
- Metrics rows: 60
- Blank responses: 0
- Unique responses: 60
- Baseline responses used the Qwen tokenizer chat template and looked structurally valid.
- Obvious baseline failure patterns included constraint misses, format non-compliance, and long refusal-behavior answers.

## Main Observations

- Overall rule-based category score increased from 0.7889 to 0.8313.
- Overall format compliance increased from 0.6667 to 0.7833.
- Average response length decreased from 28.93 to 19.63 words.
- Overall must-include coverage decreased slightly from 0.7056 to 0.6917.
- Instruction-following category score decreased from 0.9042 to 0.8000.

## Key Regressions

- Instruction-following format compliance dropped from 0.8000 to 0.5000.
- Instruction-following category score dropped from 0.9042 to 0.8000.
- Instruction-following must-not-include violation rate increased from 0.1000 to 0.2000.
- Format-compliance keyword/must-include coverage dropped from 0.8000 to 0.7000.
- Robustness keyword coverage dropped from 0.7000 to 0.6000.

## Key Improvements

These are improvements on this 60-prompt rule-based suite only.

- Format-compliance category score increased from 0.6500 to 0.8000.
- Refusal-behavior category score increased from 0.6625 to 0.7875.
- Multilingual category score increased from 0.8667 to 0.9125.
- Candidate responses were shorter on average overall.
- Candidate failure counts decreased for format non-compliance, constraint ignored, length bias, low usefulness, and under-refusal heuristics.

## Limitations

- This is a small synthetic evaluation suite, not a benchmark.
- Metrics are rule-based and can disagree with human judgment.
- No LLM-as-judge or human preference scoring was used.
- The SFT source data is broad public chat data, not targeted reliability data.
- UltraChat is synthetic/generated and may contain quality artifacts.
- Only one random seed and one small model were tested.
- The adapter was evaluated directly with PEFT; merged-model equivalence was not separately checked.

## Next Experiments

- Add targeted SFT data for exact-output extraction, JSON/regex formats, and exact-count list constraints.
- Run the same evaluation with a held-out data seed and a second decode sanity pass.
- Add perturbation pairs where the expected answer should stay invariant.
- Add a small human-review rubric for the qualitative examples.
- Test whether lowering learning rate or using fewer UltraChat examples preserves base instruction-following better.

## Commands

```powershell
.\.venv-gpu\Scripts\python.exe -m ruff check .
.\.venv-gpu\Scripts\python.exe -m pytest
.\.venv-gpu\Scripts\python.exe scripts/check_hidden_unicode.py --root .
.\.venv-gpu\Scripts\python.exe scripts/build_eval_suite.py --suite-dir eval_suites --output outputs/eval_suites/real_eval_suite.jsonl --manifest outputs/eval_suites/real_eval_suite_manifest.json
.\.venv-gpu\Scripts\python.exe scripts/validate_sft_dataset.py --dataset data/processed/qwen2_5_0_5b_sft_train.jsonl --eval-suite outputs/eval_suites/real_eval_suite.jsonl --report reports/real_runs/qwen2_5_0_5b_lora_sft_run_001/data_validation_report.md
.\.venv-gpu\Scripts\python.exe scripts/preflight_sft_format.py --model-config configs/model.yaml --sft-config configs/sft_real.yaml --dataset data/processed/qwen2_5_0_5b_sft_train.jsonl
.\.venv-gpu\Scripts\python.exe scripts/run_eval.py --config configs/eval_real.yaml --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval
.\.venv-gpu\Scripts\python.exe scripts/train_sft.py --model-config configs/model.yaml --sft-config configs/sft_real.yaml --mlflow-config configs/mlflow.yaml
.\.venv-gpu\Scripts\python.exe scripts/compare_models.py --baseline Qwen/Qwen2.5-0.5B-Instruct --candidate outputs/qwen2_5_0_5b_lora_sft_run_001/sft --backend hf --config configs/eval_real.yaml --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/comparison --reuse-baseline-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval
.\.venv-gpu\Scripts\python.exe scripts/create_real_run_report.py --run-dir reports/real_runs/qwen2_5_0_5b_lora_sft_run_001 --comparison-dir outputs/qwen2_5_0_5b_lora_sft_run_001/comparison --baseline-eval-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval --sft-output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/sft
```
