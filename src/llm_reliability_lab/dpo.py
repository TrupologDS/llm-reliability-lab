"""DPO-style preference optimization."""

from __future__ import annotations

from typing import Any

from llm_reliability_lab.config import DPOConfig, MLflowConfig, ModelConfig
from llm_reliability_lab.data import load_jsonl_dataset
from llm_reliability_lab.mlflow_utils import MLflowRun
from llm_reliability_lab.prompts import build_preference_prompt
from llm_reliability_lab.reporting import render_training_report, write_report
from llm_reliability_lab.sft import _is_number, _model_loading_kwargs
from llm_reliability_lab.utils import ensure_dir, set_seed, write_json


def format_dpo_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Format normalized preference records for TRL DPOTrainer."""

    return [
        {
            "id": record["id"],
            "prompt": build_preference_prompt(record),
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        }
        for record in records
    ]


def train_dpo(
    *,
    model_config: ModelConfig,
    dpo_config: DPOConfig,
    mlflow_config: MLflowConfig | None = None,
) -> dict[str, Any]:
    """Run DPO-style preference optimization.

    This function requires real training dependencies and compatible GPU hardware for useful runs.
    """

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer
    except ImportError as exc:
        msg = "Install training dependencies with: pip install -e '.[train]'"
        raise RuntimeError(msg) from exc

    set_seed(dpo_config.seed)
    output_dir = ensure_dir(dpo_config.output_dir)
    records = load_jsonl_dataset(dpo_config.preference_dataset_path, "preference")
    formatted = format_dpo_records(records)
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.1, seed=dpo_config.seed) if len(dataset) > 2 else None
    train_dataset = split["train"] if split is not None else dataset
    eval_dataset = split["test"] if split is not None else None

    model_path = dpo_config.sft_model_path or model_config.model_name_or_path
    tokenizer_name = model_config.tokenizer_name_or_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=model_config.trust_remote_code,
        **_model_loading_kwargs(model_config),
    )
    peft_config = LoraConfig(
        r=dpo_config.lora_r,
        lora_alpha=dpo_config.lora_alpha,
        lora_dropout=dpo_config.lora_dropout,
        target_modules=dpo_config.target_modules or None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = TRLDPOConfig(
        output_dir=str(output_dir),
        beta=dpo_config.beta,
        learning_rate=dpo_config.learning_rate,
        per_device_train_batch_size=dpo_config.batch_size,
        per_device_eval_batch_size=dpo_config.batch_size,
        gradient_accumulation_steps=dpo_config.gradient_accumulation_steps,
        num_train_epochs=dpo_config.num_train_epochs,
        max_steps=dpo_config.max_steps,
        warmup_ratio=dpo_config.warmup_ratio,
        logging_steps=dpo_config.logging_steps,
        eval_steps=dpo_config.eval_steps,
        save_steps=dpo_config.save_steps,
        report_to=[],
        seed=dpo_config.seed,
    )

    logger = MLflowRun(mlflow_config or MLflowConfig(enabled=False), run_name="dpo")
    with logger:
        logger.log_params(
            {
                "model_path": model_path,
                "preference_dataset_path": dpo_config.preference_dataset_path,
                "beta": dpo_config.beta,
                "learning_rate": dpo_config.learning_rate,
                "lora_r": dpo_config.lora_r,
                "num_train_examples": len(train_dataset),
            }
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
        train_result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        metrics = getattr(train_result, "metrics", {}) or {}
        logger.log_metrics({key: float(value) for key, value in metrics.items() if _is_number(value)})

    summary = {
        "status": "completed",
        "output_dir": str(output_dir),
        "num_records": len(records),
        "metrics": metrics,
    }
    write_json(summary, output_dir / "dpo_summary.json")
    report_path = write_report(
        output_dir / "dpo_report.md",
        render_training_report(
            title="DPO Report",
            config=dpo_config.model_dump(),
            output_dir=str(output_dir),
            metrics=metrics,
        ),
    )
    summary["report_path"] = str(report_path)
    return summary
