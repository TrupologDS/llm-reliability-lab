"""Typed configuration loading for the lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LabConfig(BaseModel):
    """Base config with strict fields to catch YAML typos early."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(LabConfig):
    """Model and tokenizer loading options."""

    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_name_or_path: str | None = None
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "bfloat16"
    trust_remote_code: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    max_seq_length: int = Field(default=2048, gt=0)
    device_map: str | dict[str, Any] | None = "auto"
    template_backend: Literal["tokenizer_chat_template", "default_chat_template"] = "default_chat_template"
    seed: int = 42

    @model_validator(mode="after")
    def validate_quantization(self) -> ModelConfig:
        """Prevent mutually exclusive quantization flags."""

        if self.load_in_4bit and self.load_in_8bit:
            msg = "Only one of load_in_4bit/load_in_8bit can be true."
            raise ValueError(msg)
        return self


class SFTConfig(LabConfig):
    """Supervised fine-tuning options."""

    dataset_path: str
    output_dir: str = "outputs/sft"
    dataset_format: Literal["prompt_completion", "chat_messages", "legacy_text"] = "prompt_completion"
    template_backend: Literal["tokenizer_chat_template", "default_chat_template"] = "default_chat_template"
    append_eos_to_completion: bool = False
    train_on_prompt: bool = False
    response_template: str = "Assistant:"
    learning_rate: float = Field(default=2e-4, gt=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    num_train_epochs: float = Field(default=1.0, ge=0)
    max_steps: int = -1
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    logging_steps: int = Field(default=10, gt=0)
    eval_steps: int = Field(default=100, gt=0)
    save_steps: int = Field(default=100, gt=0)
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=list)
    packing: bool = False
    seed: int = 42


class DPOConfig(LabConfig):
    """Preference optimization options."""

    preference_dataset_path: str
    sft_model_path: str | None = None
    output_dir: str = "outputs/dpo"
    beta: float = Field(default=0.1, gt=0)
    learning_rate: float = Field(default=5e-5, gt=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    num_train_epochs: float = Field(default=1.0, ge=0)
    max_steps: int = -1
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    logging_steps: int = Field(default=10, gt=0)
    eval_steps: int = Field(default=100, gt=0)
    save_steps: int = Field(default=100, gt=0)
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=list)
    seed: int = 42


class EvalConfig(LabConfig):
    """Evaluation and regression configuration."""

    eval_prompts_path: str
    baseline_model: str
    candidate_model: str
    candidate_base_model: str | None = None
    template_backend: Literal["tokenizer_chat_template", "default_chat_template"] = "default_chat_template"
    max_new_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    metrics: list[str] = Field(default_factory=list)
    regression_thresholds: dict[str, float] = Field(default_factory=dict)
    report_output_path: str = "reports/eval_report.md"

    @field_validator("metrics")
    @classmethod
    def metrics_must_not_be_empty(cls, value: list[str]) -> list[str]:
        """Require at least one metric name."""

        if not value:
            msg = "At least one metric must be configured."
            raise ValueError(msg)
        return value


class ServingConfig(LabConfig):
    """vLLM serving options."""

    model_path: str
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0)
    tensor_parallel_size: int = Field(default=1, gt=0)
    max_model_len: int = Field(default=2048, gt=0)
    gpu_memory_utilization: float = Field(default=0.85, gt=0, le=1)


class MLflowConfig(LabConfig):
    """MLflow tracking options."""

    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str = "llm-reliability-lab"
    run_name: str | None = None
    log_artifacts: bool = True
    log_traces: bool = False


ConfigT = TypeVar("ConfigT", bound=LabConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        msg = f"Expected YAML object in {config_path}"
        raise ValueError(msg)
    return loaded


def load_config(path: str | Path, config_type: type[ConfigT]) -> ConfigT:
    """Load and validate a typed config."""

    return config_type.model_validate(load_yaml(path))
