"""Preflight SFT formatting and completion-only collator behavior."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_reliability_lab.config import ModelConfig, SFTConfig, load_config
from llm_reliability_lab.data import load_jsonl_dataset
from llm_reliability_lab.sft import format_sft_records

IGNORE_INDEX = -100


class PreflightError(RuntimeError):
    """Raised when the SFT formatting/collator preflight fails."""


@dataclass(frozen=True)
class LabelStats:
    """Label-mask statistics for one collated example."""

    input_tokens: int
    trainable_label_tokens: int
    masked_label_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--sft-config", default="configs/sft_real.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--num-examples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_config(args.model_config, ModelConfig)
    sft_config = load_config(args.sft_config, SFTConfig)
    dataset_path = args.dataset or sft_config.dataset_path
    run_preflight(
        model_config=model_config,
        sft_config=sft_config,
        dataset_path=dataset_path,
        num_examples=args.num_examples,
    )


def run_preflight(
    *,
    model_config: ModelConfig,
    sft_config: SFTConfig,
    dataset_path: str | Path,
    num_examples: int = 5,
) -> None:
    """Run the formatting/collator preflight on a few SFT records."""

    if sft_config.dataset_format != "prompt_completion":
        raise PreflightError("preflight_sft_format currently supports dataset_format=prompt_completion.")
    if sft_config.train_on_prompt:
        raise PreflightError("preflight_sft_format expects train_on_prompt=false.")

    try:
        from transformers import AutoTokenizer
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError as exc:
        msg = "Install training dependencies with: pip install -e '.[train]'"
        raise PreflightError(msg) from exc

    tokenizer_name = model_config.tokenizer_name_or_path or model_config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_config.trust_remote_code,
    )
    validate_tokenizer_requirements(tokenizer, sft_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_jsonl_dataset(dataset_path, "sft")[:num_examples]
    if not records:
        raise PreflightError(f"No SFT records found in {dataset_path}.")

    formatted = format_sft_records(
        records,
        dataset_format=sft_config.dataset_format,
        template_backend=sft_config.template_backend,
        tokenizer=tokenizer,
        append_eos_to_completion=sft_config.append_eos_to_completion,
    )
    texts = [record["text"] for record in formatted]
    tokenized = [tokenizer(text, add_special_tokens=False) for text in texts]
    response_template_ids = tokenizer(
        sft_config.response_template,
        add_special_tokens=False,
    )["input_ids"]
    validate_response_template_presence(tokenized, response_template_ids, sft_config.response_template)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=sft_config.response_template,
        tokenizer=tokenizer,
    )
    batch = collator(tokenized)
    stats = validate_collated_labels(batch["labels"], batch["input_ids"])

    print_preview(
        formatted=formatted,
        stats=stats,
        response_template=sft_config.response_template,
        num_examples=len(formatted),
    )


def validate_tokenizer_requirements(tokenizer: Any, sft_config: SFTConfig) -> None:
    """Fail early when config requires tokenizer features that are missing."""

    if sft_config.template_backend == "tokenizer_chat_template" and not getattr(
        tokenizer,
        "chat_template",
        None,
    ):
        raise PreflightError("tokenizer_chat_template is configured, but the tokenizer has no chat_template.")
    if sft_config.append_eos_to_completion and not getattr(tokenizer, "eos_token", None):
        raise PreflightError("append_eos_to_completion=true, but the tokenizer has no eos_token.")


def validate_response_template_presence(
    tokenized_examples: list[dict[str, Any]],
    response_template_ids: list[int],
    response_template: str,
) -> None:
    """Ensure the response-template token sequence appears in every formatted example."""

    if not response_template_ids:
        raise PreflightError(f"response_template tokenized to an empty sequence: {response_template!r}")
    missing = [
        index
        for index, example in enumerate(tokenized_examples)
        if find_subsequence(example["input_ids"], response_template_ids) is None
    ]
    if missing:
        raise PreflightError(
            f"response_template was not found in formatted/tokenized examples {missing}: {response_template!r}"
        )


def validate_collated_labels(labels: Any, input_ids: Any) -> list[LabelStats]:
    """Validate completion-only label masking and return per-example stats."""

    label_rows = to_nested_lists(labels)
    input_rows = to_nested_lists(input_ids)
    stats: list[LabelStats] = []
    for labels_row, input_row in zip(label_rows, input_rows, strict=True):
        masked = sum(1 for value in labels_row if value == IGNORE_INDEX)
        trainable = sum(1 for value in labels_row if value != IGNORE_INDEX)
        input_tokens = len(input_row)
        if trainable == 0:
            raise PreflightError("All labels are masked. The response_template may not match the formatted text.")
        if masked == 0:
            raise PreflightError("Zero labels are masked. Prompt tokens would be trained; check response_template.")
        stats.append(
            LabelStats(
                input_tokens=input_tokens,
                trainable_label_tokens=trainable,
                masked_label_tokens=masked,
            )
        )
    return stats


def print_preview(
    *,
    formatted: list[dict[str, Any]],
    stats: list[LabelStats],
    response_template: str,
    num_examples: int,
) -> None:
    """Print a readable formatting/collator preview."""

    print("SFT formatting preflight passed.")
    print(f"Examples checked: {num_examples}")
    print(f"response_template: {response_template!r}")
    for index, (record, item_stats) in enumerate(zip(formatted, stats, strict=True), start=1):
        prompt = preview_text(record["prompt"])
        completion = preview_text(record["completion"])
        print(f"\nExample {index}: {record['id']}")
        print(f"formatted prompt: {prompt}")
        print(f"completion with EOS: {completion}")
        print(f"input tokens: {item_stats.input_tokens}")
        print(f"trainable label tokens: {item_stats.trainable_label_tokens}")
        print(f"masked label tokens: {item_stats.masked_label_tokens}")


def preview_text(text: str, *, max_chars: int = 500) -> str:
    """Compact long text for terminal preview."""

    compact = text.replace("\n", "\\n")
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "...[truncated]"


def find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    """Return the first index where pattern appears in values."""

    if not pattern or len(pattern) > len(values):
        return None
    last_start = len(values) - len(pattern)
    for start in range(last_start + 1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def to_nested_lists(value: Any) -> list[list[int]]:
    """Convert tensors, arrays, or nested lists to nested Python lists."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [list(row) for row in value]


if __name__ == "__main__":
    main()
