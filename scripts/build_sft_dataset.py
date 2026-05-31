"""Build a reproducible SFT JSONL dataset for the first real Qwen LoRA run."""

from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_reliability_lab.data import write_jsonl
from llm_reliability_lab.reporting import write_report

DEFAULT_SOURCE = "HuggingFaceH4/ultrachat_200k"
DEFAULT_SPLIT = "train_sft"
DEFAULT_OUTPUT = "data/processed/qwen2_5_0_5b_sft_train.jsonl"
DEFAULT_REPORT = "reports/real_runs/qwen2_5_0_5b_lora_sft_run_001/sft_dataset_build_report.md"

DATASET_METADATA = {
    "HuggingFaceH4/ultrachat_200k": {
        "license": "MIT",
        "format": "ultrachat",
        "id_prefix": "ultrachat",
    },
    "databricks/databricks-dolly-15k": {
        "license": "CC-BY-SA-3.0",
        "format": "dolly",
        "id_prefix": "dolly",
    },
}


@dataclass(frozen=True)
class BuildStats:
    """Counters recorded in the dataset build report."""

    loaded: int
    extracted: int
    filtered: int
    duplicates_removed: int
    selected: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-examples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-chars", type=int, default=8000)
    parser.add_argument("--max-response-chars", type=int, default=8000)
    parser.add_argument("--min-response-chars", type=int, default=20)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--license", default=None)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize text for exact-match deduplication."""

    return " ".join(text.casefold().split())


def extract_first_user_assistant_pair(messages: Any) -> dict[str, str] | None:
    """Extract the first user -> assistant pair from an UltraChat-like messages list."""

    if not isinstance(messages, list):
        return None

    system_messages: list[str] = []
    user_content: str | None = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role") or message.get("from") or message.get("speaker"))
        content = clean_text(message.get("content") or message.get("value") or message.get("text"))
        if not content:
            continue
        if role == "system" and user_content is None:
            system_messages.append(content)
        elif role == "user" and user_content is None:
            user_content = content
        elif role == "assistant" and user_content is not None:
            return {
                "system": "\n\n".join(system_messages),
                "instruction": user_content,
                "response": content,
            }
    return None


def normalize_role(role: Any) -> str:
    """Normalize common chat role variants."""

    value = clean_text(role).casefold()
    if value in {"human", "user"}:
        return "user"
    if value in {"assistant", "gpt", "bot"}:
        return "assistant"
    if value == "system":
        return "system"
    return value


def clean_text(value: Any) -> str:
    """Return stripped text for string-ish values."""

    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip()


def ultrachat_record_to_sft(record: dict[str, Any], *, source_value: str) -> dict[str, Any] | None:
    """Map one UltraChat record to the repository SFT schema."""

    pair = extract_first_user_assistant_pair(record.get("messages"))
    if pair is None:
        return None
    return {
        "id": "",
        "system": pair["system"],
        "instruction": pair["instruction"],
        "input": "",
        "response": pair["response"],
        "source": source_value,
    }


def dolly_record_to_sft(record: dict[str, Any], *, source_value: str) -> dict[str, Any] | None:
    """Map one Databricks Dolly record to the repository SFT schema."""

    instruction = clean_text(record.get("instruction"))
    response = clean_text(record.get("response"))
    context = clean_text(record.get("context"))
    if not instruction or not response:
        return None

    category = clean_text(record.get("category"))
    source = f"{source_value}:{category}" if category else source_value
    return {
        "id": "",
        "system": "",
        "instruction": instruction,
        "input": context,
        "response": response,
        "source": source,
    }


def filter_examples(
    examples: list[dict[str, Any]],
    *,
    max_prompt_chars: int,
    max_response_chars: int,
    min_response_chars: int,
) -> list[dict[str, Any]]:
    """Filter empty, malformed, and very long examples."""

    filtered: list[dict[str, Any]] = []
    for example in examples:
        instruction = clean_text(example.get("instruction"))
        response = clean_text(example.get("response"))
        input_text = clean_text(example.get("input"))
        prompt_chars = len(instruction) + len(input_text)
        if not instruction or not response:
            continue
        if len(response) < min_response_chars:
            continue
        if prompt_chars > max_prompt_chars or len(response) > max_response_chars:
            continue
        filtered.append(
            {
                "id": "",
                "system": clean_text(example.get("system")),
                "instruction": instruction,
                "input": input_text,
                "response": response,
                "source": clean_text(example.get("source")),
            }
        )
    return filtered


def deduplicate_by_instruction(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first example for each normalized instruction."""

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicates = 0
    for example in examples:
        key = normalize_text(str(example["instruction"]))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(example)
    return deduped, duplicates


def deterministic_sample(examples: list[dict[str, Any]], *, num_examples: int, seed: int) -> list[dict[str, Any]]:
    """Shuffle deterministically and select a bounded number of examples."""

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:num_examples]


def assign_ids(examples: list[dict[str, Any]], *, prefix: str) -> list[dict[str, Any]]:
    """Assign stable sequential IDs after deterministic sampling."""

    output: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        with_id = dict(example)
        with_id["id"] = f"{prefix}_{index:06d}"
        output.append(with_id)
    return output


def build_sft_examples(
    raw_records: list[dict[str, Any]],
    *,
    source: str,
    split: str,
    source_name: str | None,
    num_examples: int,
    seed: int,
    max_prompt_chars: int,
    max_response_chars: int,
    min_response_chars: int,
) -> tuple[list[dict[str, Any]], BuildStats]:
    """Convert raw source records into sampled repository-format SFT examples."""

    metadata = DATASET_METADATA.get(source, {})
    dataset_format = metadata.get("format", "ultrachat")
    source_value = f"{source_name or source}:{split}"

    extracted: list[dict[str, Any]] = []
    for record in raw_records:
        if dataset_format == "dolly":
            example = dolly_record_to_sft(record, source_value=source_value)
        else:
            example = ultrachat_record_to_sft(record, source_value=source_value)
        if example is not None:
            extracted.append(example)

    filtered = filter_examples(
        extracted,
        max_prompt_chars=max_prompt_chars,
        max_response_chars=max_response_chars,
        min_response_chars=min_response_chars,
    )
    deduped, duplicates_removed = deduplicate_by_instruction(filtered)
    selected = deterministic_sample(deduped, num_examples=num_examples, seed=seed)
    with_ids = assign_ids(selected, prefix=str(metadata.get("id_prefix", "sft")))
    stats = BuildStats(
        loaded=len(raw_records),
        extracted=len(extracted),
        filtered=len(filtered),
        duplicates_removed=duplicates_removed,
        selected=len(with_ids),
    )
    return with_ids, stats


def load_source_records(source: str, split: str) -> list[dict[str, Any]]:
    """Load source records lazily from Hugging Face Datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = "Install the train dependencies to build an SFT dataset from Hugging Face."
        raise RuntimeError(msg) from exc

    dataset = load_dataset(source, split=split)
    return [dict(record) for record in dataset]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_build_report(
    *,
    source: str,
    split: str,
    license_name: str,
    output_path: Path,
    output_sha256: str,
    stats: BuildStats,
    seed: int,
    max_prompt_chars: int,
    max_response_chars: int,
    min_response_chars: int,
) -> str:
    """Render a Markdown dataset build report."""

    return f"""# SFT Dataset Build Report

## Source

- Source dataset: `{source}`
- Source split: `{split}`
- Source license: `{license_name}`
- Intended run: `qwen2_5_0_5b_lora_sft_run_001`

## Output

- Output path: `{output_path.as_posix()}`
- Output SHA256: `{output_sha256}`
- Selected examples: {stats.selected}
- Random seed: {seed}

## Build Counts

- Loaded source records: {stats.loaded}
- Extracted user/assistant or instruction/response pairs: {stats.extracted}
- Records after length/content filtering: {stats.filtered}
- Duplicate normalized instructions removed: {stats.duplicates_removed}
- Records written: {stats.selected}

## Filtering Rules

- Keep records with a non-empty instruction and response.
- For UltraChat, use the first user message and the first assistant message after it.
- For UltraChat, keep a system message only if it appears before the first user message.
- Minimum response length: {min_response_chars} characters.
- Maximum prompt length: {max_prompt_chars} characters.
- Maximum response length: {max_response_chars} characters.

## Deduplication Rules

- Deduplicate by exact normalized instruction text.
- Normalization lowercases text and collapses whitespace.
- If duplicates exist, the first retained source example is kept before deterministic shuffling.

## Train/Eval Separation

- `eval_suites/` were not used as training data.
- The validation step checks exact normalized train/eval prompt overlap before training.

## Limitations

- This builder creates single-turn SFT examples from source conversations.
- Filtering is rule-based and does not assess factuality, safety, or pedagogical quality.
- License metadata is recorded from the configured public source.
- Downstream redistribution obligations still need review before publishing derived datasets.
"""


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise SystemExit("--num-examples must be positive.")

    metadata = DATASET_METADATA.get(args.source, {})
    license_name = args.license or str(metadata.get("license", "TODO: verify source license"))
    raw_records = load_source_records(args.source, args.split)
    examples, stats = build_sft_examples(
        raw_records,
        source=args.source,
        split=args.split,
        source_name=args.source_name,
        num_examples=args.num_examples,
        seed=args.seed,
        max_prompt_chars=args.max_prompt_chars,
        max_response_chars=args.max_response_chars,
        min_response_chars=args.min_response_chars,
    )
    if not examples:
        raise SystemExit("No SFT examples were produced after filtering.")

    output_path = write_jsonl(examples, args.output)
    output_sha256 = sha256_file(output_path)
    report = render_build_report(
        source=args.source,
        split=args.split,
        license_name=license_name,
        output_path=output_path,
        output_sha256=output_sha256,
        stats=stats,
        seed=args.seed,
        max_prompt_chars=args.max_prompt_chars,
        max_response_chars=args.max_response_chars,
        min_response_chars=args.min_response_chars,
    )
    report_path = write_report(args.report, report)

    print(f"Wrote {stats.selected} SFT examples to {output_path}")
    print(f"SHA256: {output_sha256}")
    print(f"Report: {report_path}")
    if stats.selected < args.num_examples:
        print(f"Warning: requested {args.num_examples} examples but only {stats.selected} passed filters.")


if __name__ == "__main__":
    main()
