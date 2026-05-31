"""Dataset loading, schema normalization, and lightweight data validation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llm_reliability_lab.utils import ensure_dir

DatasetKind = Literal["sft", "preference", "eval"]

EVAL_CATEGORIES = {
    "instruction_following",
    "factuality",
    "format_compliance",
    "refusal_behavior",
    "multilingual",
    "reasoning",
    "robustness",
}


@dataclass(frozen=True)
class DataIssue:
    """A structured data quality issue."""

    code: str
    message: str
    severity: Literal["warning", "error"]
    record_id: str | None = None
    row_number: int | None = None


class DataValidationError(ValueError):
    """Raised when validation finds blocking data issues."""

    def __init__(self, issues: list[DataIssue]) -> None:
        self.issues = issues
        details = "; ".join(issue.message for issue in issues[:5])
        if len(issues) > 5:
            details += f"; ... and {len(issues) - 5} more"
        super().__init__(details)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""

    records: list[dict[str, Any]] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        for row_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issue = DataIssue(
                    code="INVALID_JSON",
                    message=f"{input_path}:{row_number} is not valid JSON: {exc}",
                    severity="error",
                    row_number=row_number,
                )
                raise DataValidationError([issue]) from exc
            if not isinstance(record, dict):
                issue = DataIssue(
                    code="INVALID_RECORD",
                    message=f"{input_path}:{row_number} must be a JSON object.",
                    severity="error",
                    row_number=row_number,
                )
                raise DataValidationError([issue])
            records.append(record)
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write records to JSONL with stable ASCII-safe encoding."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            file.write("\n")
    return output_path


def _as_text(record: dict[str, Any], field: str, *, default: str | None = None) -> str:
    value = record.get(field, default)
    if value is None:
        return ""
    if not isinstance(value, str):
        msg = f"Field '{field}' must be a string."
        raise ValueError(msg)
    return value.strip()


def _require_fields(record: dict[str, Any], fields: list[str], row_number: int | None) -> None:
    missing = [field for field in fields if field not in record or record[field] is None]
    if missing:
        record_id = str(record.get("id")) if record.get("id") is not None else None
        issues = [
            DataIssue(
                code="MISSING_FIELD",
                message=f"Missing required field(s): {', '.join(missing)}",
                severity="error",
                record_id=record_id,
                row_number=row_number,
            )
        ]
        raise DataValidationError(issues)


def normalize_instruction_example(
    record: dict[str, Any],
    row_number: int | None = None,
) -> dict[str, Any]:
    """Normalize one SFT/instruction example."""

    _require_fields(record, ["id", "instruction", "response", "source"], row_number)
    return {
        "id": _as_text(record, "id"),
        "system": _as_text(record, "system", default=""),
        "instruction": _as_text(record, "instruction"),
        "input": _as_text(record, "input", default=""),
        "response": _as_text(record, "response"),
        "source": _as_text(record, "source"),
    }


def normalize_preference_example(
    record: dict[str, Any],
    row_number: int | None = None,
) -> dict[str, Any]:
    """Normalize one DPO/preference example."""

    _require_fields(record, ["id", "prompt", "chosen", "rejected", "source"], row_number)
    return {
        "id": _as_text(record, "id"),
        "prompt": _as_text(record, "prompt"),
        "chosen": _as_text(record, "chosen"),
        "rejected": _as_text(record, "rejected"),
        "source": _as_text(record, "source"),
    }


def normalize_eval_constraints(raw_constraints: Any) -> dict[str, Any]:
    """Normalize evaluation constraints with conservative defaults."""

    if raw_constraints is None:
        raw_constraints = {}
    if not isinstance(raw_constraints, dict):
        msg = "Eval constraints must be an object."
        raise ValueError(msg)

    must_include = raw_constraints.get("must_include", [])
    must_not_include = raw_constraints.get("must_not_include", [])
    if not isinstance(must_include, list) or not all(isinstance(x, str) for x in must_include):
        msg = "constraints.must_include must be a list of strings."
        raise ValueError(msg)
    if not isinstance(must_not_include, list) or not all(isinstance(x, str) for x in must_not_include):
        msg = "constraints.must_not_include must be a list of strings."
        raise ValueError(msg)

    required_format = raw_constraints.get("required_format", "none") or "none"
    if required_format not in {"json", "bullets", "short_answer", "free_text", "none"}:
        msg = f"Unsupported required_format: {required_format}"
        raise ValueError(msg)

    max_words = raw_constraints.get("max_words")
    if max_words is not None and (not isinstance(max_words, int) or max_words <= 0):
        msg = "constraints.max_words must be a positive integer or null."
        raise ValueError(msg)

    regex = raw_constraints.get("regex")
    if regex is not None and not isinstance(regex, str):
        msg = "constraints.regex must be a string when provided."
        raise ValueError(msg)

    return {
        "must_include": [item.strip() for item in must_include if item.strip()],
        "must_not_include": [item.strip() for item in must_not_include if item.strip()],
        "required_format": required_format,
        "max_words": max_words,
        "regex": regex,
    }


def normalize_eval_prompt(record: dict[str, Any], row_number: int | None = None) -> dict[str, Any]:
    """Normalize one evaluation prompt."""

    _require_fields(record, ["id", "category", "prompt", "expected_behavior"], row_number)
    risk_tags = record.get("risk_tags", [])
    if risk_tags is None:
        risk_tags = []
    if not isinstance(risk_tags, list) or not all(isinstance(tag, str) for tag in risk_tags):
        msg = "risk_tags must be a list of strings."
        raise ValueError(msg)
    return {
        "id": _as_text(record, "id"),
        "category": _as_text(record, "category"),
        "prompt": _as_text(record, "prompt"),
        "expected_behavior": _as_text(record, "expected_behavior"),
        "constraints": normalize_eval_constraints(record.get("constraints", {})),
        "risk_tags": [tag.strip() for tag in risk_tags if tag.strip()],
    }


def normalize_records(records: list[dict[str, Any]], kind: DatasetKind) -> list[dict[str, Any]]:
    """Normalize records for a specific dataset kind."""

    normalizers = {
        "sft": normalize_instruction_example,
        "preference": normalize_preference_example,
        "eval": normalize_eval_prompt,
    }
    normalizer = normalizers[kind]
    normalized: list[dict[str, Any]] = []
    issues: list[DataIssue] = []
    for row_number, record in enumerate(records, start=1):
        try:
            normalized.append(normalizer(record, row_number))
        except (DataValidationError, ValueError) as exc:
            if isinstance(exc, DataValidationError):
                issues.extend(exc.issues)
            else:
                issues.append(
                    DataIssue(
                        code="SCHEMA_ERROR",
                        message=str(exc),
                        severity="error",
                        record_id=str(record.get("id")) if record.get("id") is not None else None,
                        row_number=row_number,
                    )
                )
    if issues:
        raise DataValidationError(issues)
    return normalized


def detect_data_issues(
    records: list[dict[str, Any]],
    kind: DatasetKind,
    *,
    max_text_chars: int = 16_000,
) -> list[DataIssue]:
    """Detect blocking and non-blocking data quality issues."""

    issues: list[DataIssue] = []
    seen_ids: dict[str, int] = {}
    seen_prompts: dict[str, int] = {}

    for row_number, record in enumerate(records, start=1):
        record_id = str(record.get("id", "")).strip() or None
        if not record_id:
            issues.append(DataIssue("EMPTY_ID", "Record id is empty.", "error", None, row_number))
        elif record_id in seen_ids:
            issues.append(
                DataIssue(
                    "DUPLICATE_ID",
                    f"Duplicate id '{record_id}' also appears at row {seen_ids[record_id]}.",
                    "error",
                    record_id,
                    row_number,
                )
            )
        else:
            seen_ids[record_id] = row_number

        prompt = _prompt_for_duplicate_detection(record, kind)
        if not prompt:
            issues.append(DataIssue("EMPTY_PROMPT", "Prompt/instruction is empty.", "error", record_id, row_number))
        elif prompt in seen_prompts:
            issues.append(
                DataIssue(
                    "DUPLICATE_PROMPT",
                    f"Duplicate prompt also appears at row {seen_prompts[prompt]}.",
                    "warning",
                    record_id,
                    row_number,
                )
            )
        else:
            seen_prompts[prompt] = row_number

        joined_text = "\n".join(str(value) for value in record.values())
        if len(joined_text) > max_text_chars:
            issues.append(
                DataIssue(
                    "LONG_EXAMPLE",
                    f"Example has {len(joined_text)} characters.",
                    "warning",
                    record_id,
                    row_number,
                )
            )

        if kind == "sft" and not str(record.get("response", "")).strip():
            issues.append(DataIssue("EMPTY_RESPONSE", "SFT response is empty.", "error", record_id, row_number))
        if kind == "preference":
            chosen = str(record.get("chosen", "")).strip()
            rejected = str(record.get("rejected", "")).strip()
            if not chosen or not rejected:
                issues.append(
                    DataIssue(
                        "EMPTY_PREFERENCE_RESPONSE",
                        "Preference chosen/rejected response is empty.",
                        "error",
                        record_id,
                        row_number,
                    )
                )
            if chosen and chosen == rejected:
                issues.append(
                    DataIssue(
                        "IDENTICAL_PREFERENCE_PAIR",
                        "Preference chosen and rejected responses are identical.",
                        "error",
                        record_id,
                        row_number,
                    )
                )
        if kind == "eval":
            category = str(record.get("category", "")).strip()
            if not category:
                issues.append(
                    DataIssue("MISSING_CATEGORY", "Eval category is missing.", "error", record_id, row_number)
                )
            elif category not in EVAL_CATEGORIES:
                issues.append(
                    DataIssue(
                        "UNSUPPORTED_CATEGORY",
                        f"Unsupported eval category '{category}'.",
                        "error",
                        record_id,
                        row_number,
                    )
                )

    return issues


def _prompt_for_duplicate_detection(record: dict[str, Any], kind: DatasetKind) -> str:
    if kind == "sft":
        return "\n".join(
            part.strip() for part in [str(record.get("instruction", "")), str(record.get("input", ""))] if part.strip()
        )
    return str(record.get("prompt", "")).strip()


def validate_records(records: list[dict[str, Any]], kind: DatasetKind) -> list[dict[str, Any]]:
    """Normalize records and raise if blocking issues are found."""

    normalized = normalize_records(records, kind)
    issues = detect_data_issues(normalized, kind)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise DataValidationError(errors)
    return normalized


def load_jsonl_dataset(path: str | Path, kind: DatasetKind) -> list[dict[str, Any]]:
    """Load and validate a JSONL dataset."""

    return validate_records(read_jsonl(path), kind)


def load_hf_dataset(
    dataset_name: str,
    *,
    kind: DatasetKind,
    split: str = "train",
    config_name: str | None = None,
    **load_kwargs: Any,
) -> list[dict[str, Any]]:
    """Load a public Hugging Face dataset split and validate records.

    This function is optional and imports `datasets` lazily so tests do not require network access.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = "Install the train extra to use Hugging Face dataset loading."
        raise RuntimeError(msg) from exc

    dataset = load_dataset(dataset_name, config_name, split=split, **load_kwargs)
    return validate_records([dict(record) for record in dataset], kind)


def to_hf_dataset(records: list[dict[str, Any]]):
    """Convert normalized records to a Hugging Face Dataset lazily."""

    try:
        from datasets import Dataset
    except ImportError as exc:
        msg = "Install the train extra to convert records to a Hugging Face Dataset."
        raise RuntimeError(msg) from exc
    return Dataset.from_list(records)


def split_records(
    records: list[dict[str, Any]],
    *,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split records into train/validation/test."""

    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-9:
        msg = "train_size + validation_size + test_size must equal 1.0"
        raise ValueError(msg)

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_records = len(shuffled)
    train_end = int(n_records * train_size)
    validation_end = train_end + int(n_records * validation_size)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
