# LLM Reliability Lab

A reproducible research-engineering lab for post-training, evaluating and debugging small open-weight language models.

The lab focuses on practical reliability questions that appear after supervised fine-tuning or preference optimization: format regressions, over-refusal, under-refusal, length bias, prompt brittleness, unsupported claims, and degraded multilingual behavior.

No model weights, checkpoints, private datasets, generated logs, API keys, Hugging Face caches, or unsupported benchmark claims are committed to this repository.

## What Is Included

- Config-driven SFT with LoRA/PEFT.
- Config-driven DPO-style preference optimization.
- Baseline vs candidate evaluation.
- Rule-based behavior metrics and regression detection.
- Failure taxonomy for qualitative error analysis.
- MLflow helpers for local experiment tracking.
- Optional vLLM serving helper.
- Report templates, sample pipeline reports, and real-run documentation.

## Sample Checks vs Real Experiments vs TODOs

Sample pipeline checks:

- Use tiny synthetic data in `data/samples/`.
- Use the deterministic `mock` backend.
- Validate data loading, metrics, reporting, and CI plumbing.
- Write outputs under `outputs/` and generated reports under `reports/sample/`.
- Are not benchmark results.

Real model experiments:

- Use real model paths or Hugging Face model IDs.
- Use `--backend hf` or `--backend vllm`.
- Save baseline and candidate outputs, metrics, regressions, configs, commands, and qualitative examples.
- Store reports under `reports/real_runs/<run_name>/`.

TODOs:

- Remain in templates and reports until real experiments are run.
- Must not be replaced with guessed metrics or unsupported conclusions.

## Architecture

```text
Data preparation
  -> Baseline evaluation
  -> SFT
  -> DPO-style preference optimization
  -> Candidate evaluation
  -> Regression detection
  -> Error analysis
  -> Reports/model card
```

## Repository Structure

```text
configs/                 YAML configs for model, SFT, DPO, evaluation, serving, and MLflow
data/samples/            Tiny synthetic fixtures for tests and sample checks
eval_suites/             Synthetic evaluation prompt suites, not training data
src/llm_reliability_lab/ Package code
scripts/                 CLI entry points
reports/sample/          Generated sample pipeline reports
reports/templates/       Reusable report templates
reports/real_runs/       Real experiment report area
tests/                   CPU-only tests
.github/workflows/       CI
```

## Compatibility

The training stack is pinned to a conservative TRL 0.9-style API range.

| Component | Target range | Notes |
| --- | --- | --- |
| Python | `>=3.11` | CI uses Python 3.11; local verification used Python 3.12. |
| PyTorch | `>=2.3,<2.5` | Required for real HF training/inference. |
| Transformers | `>=4.43,<4.46` | Used by SFT/DPO/HF inference paths. |
| TRL | `>=0.9.6,<0.10` | Uses `SFTTrainer`, `DPOTrainer`, and completion-only data collator. |
| PEFT | `>=0.11,<0.13` | LoRA adapter configuration. |
| CPU mode | tested | Unit tests and sample mock evaluation. |
| GPU mode | tested for Run 001 | Qwen2.5-0.5B LoRA SFT completed on RTX 5090; see [Run 001](reports/real_runs/qwen2_5_0_5b_lora_sft_run_001/). |

## Quickstart

```bash
make setup
make lint
make test
make check-hidden-unicode
make prepare-sample-data
make evaluate-sample
make report-sample
```

If `make` is unavailable:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
python scripts/check_hidden_unicode.py --root .
python scripts/prepare_data.py --input-dir data/samples --output-dir outputs/sample_data
python scripts/run_eval.py --config configs/eval.yaml --backend mock --model sample-pipeline-check --output-dir outputs/sample_eval
python scripts/generate_report.py --eval-summary outputs/sample_eval/eval_summary.json --output-dir reports/sample
```

## Baseline vs Candidate Evaluation

Use `scripts/compare_models.py` to evaluate two model versions with the same prompt suite and decoding settings:

```bash
python scripts/compare_models.py \
  --baseline Qwen/Qwen2.5-0.5B-Instruct \
  --candidate outputs/dpo \
  --backend hf \
  --config configs/eval.yaml \
  --output-dir outputs/comparisons/qwen_dpo_run_001
```

The script writes:

- baseline generations and metrics
- candidate generations and metrics
- regression records
- regression report
- qualitative examples for review

## First Real Experiment Plan

The first real run compares `Qwen/Qwen2.5-0.5B-Instruct` against a LoRA SFT candidate trained from `configs/sft_real.yaml`.

Sample reports are plumbing checks only. Real-run reports under `reports/real_runs/` are the only evidence of model behavior. No improvement is claimed until the run is completed, reviewed, and documented.

Install the training and evaluation extras:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -e ".[train,eval,dev]"
```

Build the merged real evaluation suite from `eval_suites/`:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/build_eval_suite.py --suite-dir eval_suites --output outputs/eval_suites/real_eval_suite.jsonl --manifest outputs/eval_suites/real_eval_suite_manifest.json
```

Build the SFT training dataset from UltraChat:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/build_sft_dataset.py --source HuggingFaceH4/ultrachat_200k --split train_sft --num-examples 3000 --seed 42 --output data/processed/qwen2_5_0_5b_sft_train.jsonl
```

The default first-run source is `HuggingFaceH4/ultrachat_200k` with the `train_sft` split. The builder extracts single-turn user/assistant pairs, filters malformed or very long examples, deduplicates by normalized instruction, samples deterministically, and writes a build report with the output SHA256. `eval_suites/` are not used as training data.

Validate the SFT dataset before using GPU time:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/validate_sft_dataset.py --dataset data/processed/qwen2_5_0_5b_sft_train.jsonl --eval-suite outputs/eval_suites/real_eval_suite.jsonl --report reports/real_runs/qwen2_5_0_5b_lora_sft_run_001/data_validation_report.md
```

Preflight SFT formatting and completion-only masking:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/preflight_sft_format.py --model-config configs/model.yaml --sft-config configs/sft_real.yaml --dataset data/processed/qwen2_5_0_5b_sft_train.jsonl
```

This loads the configured Qwen tokenizer, applies the same SFT formatter used by training, instantiates `DataCollatorForCompletionOnlyLM`, and verifies that prompt tokens are masked while assistant completion tokens remain trainable.

Run baseline evaluation:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_eval.py --config configs/eval_real.yaml --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval
```

Train the LoRA SFT candidate:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/train_sft.py --model-config configs/model.yaml --sft-config configs/sft_real.yaml --mlflow-config configs/mlflow.yaml
```

The SFT output is a PEFT/LoRA adapter directory by default. There are two supported candidate evaluation paths:

Evaluate the adapter directly:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/compare_models.py --baseline Qwen/Qwen2.5-0.5B-Instruct --candidate outputs/qwen2_5_0_5b_lora_sft_run_001/sft --backend hf --config configs/eval_real.yaml --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/comparison --reuse-baseline-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval
```

This loads `configs/eval_real.yaml`, detects `adapter_config.json`, loads `candidate_base_model`, and attaches the adapter with PEFT before generation.

Or merge the adapter into a standalone model first:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/merge_lora_adapter.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qwen2_5_0_5b_lora_sft_run_001/sft --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/sft_merged
.\.venv-gpu\Scripts\python.exe scripts/compare_models.py --baseline Qwen/Qwen2.5-0.5B-Instruct --candidate outputs/qwen2_5_0_5b_lora_sft_run_001/sft_merged --backend hf --config configs/eval_real.yaml --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/comparison_merged --reuse-baseline-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval
```

Generate or refresh the real-run report folder:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/create_real_run_report.py --run-dir reports/real_runs/qwen2_5_0_5b_lora_sft_run_001 --comparison-dir outputs/qwen2_5_0_5b_lora_sft_run_001/comparison --baseline-eval-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval --sft-output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/sft
```

The report is written to:

```text
reports/real_runs/qwen2_5_0_5b_lora_sft_run_001/
```

The report command leaves missing sections marked TODO and reads only existing run outputs.

## First Real Run Summary

`reports/real_runs/qwen2_5_0_5b_lora_sft_run_001/` contains the first completed local run: Qwen2.5 0.5B baseline vs a LoRA SFT adapter trained on 3,000 examples sampled from `HuggingFaceH4/ultrachat_200k` (`train_sft`, MIT license). The run used the 60-prompt synthetic reliability suite in `eval_suites/`.

The reports show mixed behavior: rule-based overall score and format compliance improved on this small suite, while strict instruction-following regressed on exact-output and length/count constraints. Treat these results as a reproducible debugging artifact, not a benchmark claim.

## Real Training Workflow

Real SFT/DPO runs require compatible GPU hardware and the training extra:

```bash
python -m pip install -e ".[train,eval,dev]"
make train-sft
make train-dpo
make evaluate
make report
```

Model choice is configured in `configs/model.yaml`. Suggested small open-weight models include:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Evaluation Suites

`eval_suites/` contains synthetic evaluation prompts grouped by behavior:

- instruction following
- format compliance
- multilingual behavior
- refusal behavior
- grounded QA
- robustness perturbations

These are evaluation prompts only. They are not training data and are not comprehensive benchmarks.

## How to Add a Real Experiment

1. Create or copy a config set for the run.
2. Record exact model revisions, dataset versions, and hardware.
3. Run baseline evaluation.
4. Run SFT and/or DPO if needed.
5. Run candidate evaluation with the same eval suite and decoding settings.
6. Run `scripts/compare_models.py`.
7. Write a report under `reports/real_runs/<run_name>/`.
8. Fill in model card, data card, regression report, qualitative examples, limitations, and next experiments.
9. Keep raw generated outputs in ignored artifact storage or an external experiment tracker, not in git.

See `reports/real_runs/README.md` for the expected run contents.

## Data Formats

SFT JSONL:

```json
{"id":"example_001","system":"optional system prompt","instruction":"user instruction","input":"optional input/context","response":"assistant response","source":"dataset/source name"}
```

Preference JSONL:

```json
{"id":"pref_001","prompt":"user prompt","chosen":"preferred answer","rejected":"rejected answer","source":"dataset/source name"}
```

Evaluation JSONL:

```json
{"id":"eval_001","category":"format_compliance","prompt":"test prompt","expected_behavior":"short description","constraints":{"must_include":[],"must_not_include":[],"required_format":"json","max_words":null},"risk_tags":[]}
```

## Limitations

- Small models are not frontier models.
- Project-local eval suites are not comprehensive benchmarks.
- Rule-based metrics can miss semantic failures.
- LLM-as-judge evaluation is optional and disabled by default.
- Real conclusions require fixed configs, comparable baselines, qualitative review, and documented hardware.

## Roadmap

- Add perturbation generation.
- Add lm-evaluation-harness integration.
- Add RAG and agent evaluation.
- Add human-review annotation workflow.
- Add distributed training configs.
