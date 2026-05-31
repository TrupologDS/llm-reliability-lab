PYTHON ?= python
SFT_REAL_DATASET ?= data/samples/sft_sample.jsonl

.PHONY: setup lint format test check-hidden-unicode prepare-sample-data evaluate-sample report-sample build-real-eval-suite validate-sft-real preflight-sft-format eval-baseline train-sft-real merge-sft-lora compare-baseline-sft compare-baseline-sft-merged real-run-report train-sft train-dpo evaluate report clean

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

check-hidden-unicode:
	$(PYTHON) scripts/check_hidden_unicode.py --root .

prepare-sample-data:
	$(PYTHON) scripts/prepare_data.py --input-dir data/samples --output-dir outputs/sample_data

evaluate-sample:
	$(PYTHON) scripts/run_eval.py --config configs/eval.yaml --backend mock --model mock-sample-model --output-dir outputs/sample_eval

report-sample:
	$(PYTHON) scripts/generate_report.py --eval-summary outputs/sample_eval/eval_summary.json --output-dir reports/sample

build-real-eval-suite:
	$(PYTHON) scripts/build_eval_suite.py --suite-dir eval_suites --output outputs/eval_suites/real_eval_suite.jsonl --manifest outputs/eval_suites/real_eval_suite_manifest.json

validate-sft-real:
	$(PYTHON) scripts/validate_sft_dataset.py --dataset $(SFT_REAL_DATASET) --eval-suite outputs/eval_suites/real_eval_suite.jsonl --report reports/real_runs/qwen2_5_0_5b_lora_sft_run_001/data_validation_report.md

preflight-sft-format:
	$(PYTHON) scripts/preflight_sft_format.py --model-config configs/model.yaml --sft-config configs/sft_real.yaml --dataset $(SFT_REAL_DATASET)

eval-baseline:
	$(PYTHON) scripts/run_eval.py --config configs/eval_real.yaml --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval

train-sft-real:
	$(PYTHON) scripts/train_sft.py --model-config configs/model.yaml --sft-config configs/sft_real.yaml --mlflow-config configs/mlflow.yaml

merge-sft-lora:
	$(PYTHON) scripts/merge_lora_adapter.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qwen2_5_0_5b_lora_sft_run_001/sft --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/sft_merged

compare-baseline-sft:
	$(PYTHON) scripts/compare_models.py --baseline Qwen/Qwen2.5-0.5B-Instruct --candidate outputs/qwen2_5_0_5b_lora_sft_run_001/sft --backend hf --config configs/eval_real.yaml --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/comparison --reuse-baseline-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval

compare-baseline-sft-merged:
	$(PYTHON) scripts/compare_models.py --baseline Qwen/Qwen2.5-0.5B-Instruct --candidate outputs/qwen2_5_0_5b_lora_sft_run_001/sft_merged --backend hf --config configs/eval_real.yaml --output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/comparison_merged --reuse-baseline-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval

real-run-report:
	$(PYTHON) scripts/create_real_run_report.py --run-dir reports/real_runs/qwen2_5_0_5b_lora_sft_run_001 --comparison-dir outputs/qwen2_5_0_5b_lora_sft_run_001/comparison --baseline-eval-dir outputs/qwen2_5_0_5b_lora_sft_run_001/baseline_eval --sft-output-dir outputs/qwen2_5_0_5b_lora_sft_run_001/sft

train-sft:
	$(PYTHON) scripts/train_sft.py --model-config configs/model.yaml --sft-config configs/sft.yaml --mlflow-config configs/mlflow.yaml

train-dpo:
	$(PYTHON) scripts/train_dpo.py --model-config configs/model.yaml --dpo-config configs/dpo.yaml --mlflow-config configs/mlflow.yaml

evaluate:
	$(PYTHON) scripts/run_eval.py --config configs/eval.yaml --backend hf --output-dir outputs/eval

report:
	$(PYTHON) scripts/generate_report.py --eval-summary outputs/eval/eval_summary.json --output-dir reports/real_runs/latest

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['outputs', '.pytest_cache', '.ruff_cache']]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
