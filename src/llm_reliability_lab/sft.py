"""Supervised fine-tuning with LoRA/PEFT."""

from __future__ import annotations

from typing import Any

from llm_reliability_lab.config import MLflowConfig, ModelConfig, SFTConfig
from llm_reliability_lab.data import load_jsonl_dataset
from llm_reliability_lab.mlflow_utils import MLflowRun
from llm_reliability_lab.prompts import (
    ChatTemplateTokenizer,
    build_sft_messages,
    format_messages,
    format_sft_text,
    messages_to_dicts,
)
from llm_reliability_lab.reporting import render_training_report, write_report
from llm_reliability_lab.utils import ensure_dir, set_seed, write_json


def format_sft_records(
    records: list[dict[str, Any]],
    *,
    dataset_format: str = "prompt_completion",
    template_backend: str = "default_chat_template",
    tokenizer: ChatTemplateTokenizer | None = None,
    append_eos_to_completion: bool = False,
) -> list[dict[str, Any]]:
    """Format normalized SFT records for instruction tuning.

    `prompt_completion` is the default because it keeps the supervised target explicit.
    `chat_messages` preserves role-structured data for chat-template-aware tooling.
    `legacy_text` keeps the original single-text path for simple experiments.
    """

    formatters = {
        "prompt_completion": format_prompt_completion_record,
        "chat_messages": format_chat_messages_record,
        "legacy_text": format_legacy_text_record,
    }
    if dataset_format not in formatters:
        msg = f"Unsupported SFT dataset_format: {dataset_format}"
        raise ValueError(msg)
    return [
        formatters[dataset_format](
            record,
            template_backend=template_backend,
            tokenizer=tokenizer,
            append_eos_to_completion=append_eos_to_completion,
        )
        for record in records
    ]


def format_prompt_completion_record(
    record: dict[str, Any],
    *,
    template_backend: str = "default_chat_template",
    tokenizer: ChatTemplateTokenizer | None = None,
    append_eos_to_completion: bool = False,
) -> dict[str, Any]:
    """Return prompt/completion fields plus a compatibility text field."""

    prompt_messages = build_sft_messages(record, include_response=False)
    prompt = format_messages(
        prompt_messages,
        template_backend=template_backend,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        add_generation_prompt=True,
    )
    completion = str(record.get("response", "")).strip()
    if append_eos_to_completion:
        completion = append_eos_once(completion, tokenizer)
    return {
        "id": record["id"],
        "prompt": prompt,
        "completion": completion,
        "text": f"{prompt} {completion}".strip(),
    }


def format_chat_messages_record(
    record: dict[str, Any],
    *,
    template_backend: str = "default_chat_template",
    tokenizer: ChatTemplateTokenizer | None = None,
    append_eos_to_completion: bool = False,
) -> dict[str, Any]:
    """Return OpenAI-style chat messages plus a compatibility text field."""

    messages = build_sft_messages(record, include_response=True)
    _ = append_eos_to_completion
    return {
        "id": record["id"],
        "messages": messages_to_dicts(messages),
        "text": format_messages(
            messages,
            template_backend=template_backend,  # type: ignore[arg-type]
            tokenizer=tokenizer,
        ),
    }


def format_legacy_text_record(
    record: dict[str, Any],
    *,
    template_backend: str = "default_chat_template",
    tokenizer: ChatTemplateTokenizer | None = None,
    append_eos_to_completion: bool = False,
) -> dict[str, Any]:
    """Return the original single text field."""

    _ = append_eos_to_completion
    return {
        "id": record["id"],
        "text": format_sft_text(
            record,
            template_backend=template_backend,  # type: ignore[arg-type]
            tokenizer=tokenizer,
        ),
    }


def train_sft(
    *,
    model_config: ModelConfig,
    sft_config: SFTConfig,
    mlflow_config: MLflowConfig | None = None,
) -> dict[str, Any]:
    """Run LoRA SFT.

    This function requires the `train` extra and, for practical runs, GPU hardware.
    """

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig as TRLSFTConfig
        from trl import SFTTrainer
    except ImportError as exc:
        msg = "Install training dependencies with: pip install -e '.[train]'"
        raise RuntimeError(msg) from exc

    set_seed(sft_config.seed)
    output_dir = ensure_dir(sft_config.output_dir)
    records = load_jsonl_dataset(sft_config.dataset_path, "sft")

    tokenizer_name = model_config.tokenizer_name_or_path or model_config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted = format_sft_records(
        records,
        dataset_format=sft_config.dataset_format,
        template_backend=sft_config.template_backend,
        tokenizer=tokenizer,
        append_eos_to_completion=sft_config.append_eos_to_completion,
    )
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.1, seed=sft_config.seed) if len(dataset) > 2 else None
    train_dataset = split["train"] if split is not None else dataset
    eval_dataset = split["test"] if split is not None else None

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **_model_loading_kwargs(model_config),
    )

    peft_config = LoraConfig(
        r=sft_config.lora_r,
        lora_alpha=sft_config.lora_alpha,
        lora_dropout=sft_config.lora_dropout,
        target_modules=sft_config.target_modules or None,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TRLSFTConfig(
        output_dir=str(output_dir),
        learning_rate=sft_config.learning_rate,
        per_device_train_batch_size=sft_config.batch_size,
        per_device_eval_batch_size=sft_config.batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        num_train_epochs=sft_config.num_train_epochs,
        max_steps=sft_config.max_steps,
        warmup_ratio=sft_config.warmup_ratio,
        logging_steps=sft_config.logging_steps,
        eval_steps=sft_config.eval_steps,
        save_steps=sft_config.save_steps,
        max_seq_length=model_config.max_seq_length,
        packing=sft_config.packing,
        dataset_text_field="text",
        report_to=[],
        seed=sft_config.seed,
    )
    data_collator = None
    if not sft_config.train_on_prompt:
        try:
            from trl import DataCollatorForCompletionOnlyLM
        except ImportError as exc:
            msg = (
                "train_on_prompt=false requires TRL DataCollatorForCompletionOnlyLM. "
                "Use the pinned train extra or set train_on_prompt=true."
            )
            raise RuntimeError(msg) from exc
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=sft_config.response_template,
            tokenizer=tokenizer,
        )

    logger = MLflowRun(mlflow_config or MLflowConfig(enabled=False), run_name="sft")
    with logger:
        logger.log_params(
            {
                "model_name_or_path": model_config.model_name_or_path,
                "dataset_path": sft_config.dataset_path,
                "learning_rate": sft_config.learning_rate,
                "dataset_format": sft_config.dataset_format,
                "template_backend": sft_config.template_backend,
                "append_eos_to_completion": sft_config.append_eos_to_completion,
                "train_on_prompt": sft_config.train_on_prompt,
                "lora_r": sft_config.lora_r,
                "lora_alpha": sft_config.lora_alpha,
                "lora_dropout": sft_config.lora_dropout,
                "num_train_examples": len(train_dataset),
            }
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            data_collator=data_collator,
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
    write_json(summary, output_dir / "sft_summary.json")
    report_path = write_report(
        output_dir / "sft_report.md",
        render_training_report(
            title="SFT Report",
            config=sft_config.model_dump(),
            output_dir=str(output_dir),
            metrics=metrics,
        ),
    )
    summary["report_path"] = str(report_path)
    return summary


def _model_loading_kwargs(model_config: ModelConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"device_map": model_config.device_map}
    if model_config.torch_dtype != "auto":
        try:
            import torch
        except ImportError as exc:
            msg = "PyTorch is required for model loading."
            raise RuntimeError(msg) from exc
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        kwargs["torch_dtype"] = dtype_map[model_config.torch_dtype]
    else:
        kwargs["torch_dtype"] = "auto"

    if model_config.load_in_4bit or model_config.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            msg = "BitsAndBytes quantization requires transformers and bitsandbytes."
            raise RuntimeError(msg) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=model_config.load_in_4bit,
            load_in_8bit=model_config.load_in_8bit,
        )
    return kwargs


def append_eos_once(completion: str, tokenizer: ChatTemplateTokenizer | None) -> str:
    """Append tokenizer.eos_token to completion if it is not already present."""

    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_token:
        msg = "append_eos_to_completion=true requires a tokenizer with eos_token."
        raise ValueError(msg)
    if completion.endswith(eos_token):
        return completion
    return f"{completion}{eos_token}"


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
