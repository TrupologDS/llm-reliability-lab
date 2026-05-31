from __future__ import annotations

import pytest

from llm_reliability_lab.data import (
    DataValidationError,
    detect_data_issues,
    load_jsonl_dataset,
    validate_records,
)


def test_load_sample_sft_dataset() -> None:
    records = load_jsonl_dataset("data/samples/sft_sample.jsonl", "sft")
    assert len(records) >= 5
    assert records[0]["instruction"]
    assert "system" in records[0]


def test_duplicate_ids_are_blocking() -> None:
    records = [
        {"id": "x", "instruction": "Do a", "response": "A", "source": "test"},
        {"id": "x", "instruction": "Do b", "response": "B", "source": "test"},
    ]
    with pytest.raises(DataValidationError) as exc_info:
        validate_records(records, "sft")
    assert any(issue.code == "DUPLICATE_ID" for issue in exc_info.value.issues)


def test_duplicate_prompts_are_detected_as_warning() -> None:
    records = [
        {"id": "a", "instruction": "Same", "response": "A", "source": "test", "system": "", "input": ""},
        {"id": "b", "instruction": "Same", "response": "B", "source": "test", "system": "", "input": ""},
    ]
    normalized = validate_records(records, "sft")
    issues = detect_data_issues(normalized, "sft")
    assert any(issue.code == "DUPLICATE_PROMPT" and issue.severity == "warning" for issue in issues)


def test_preference_chosen_rejected_must_differ() -> None:
    records = [
        {
            "id": "pref",
            "prompt": "Choose",
            "chosen": "same",
            "rejected": "same",
            "source": "test",
        }
    ]
    with pytest.raises(DataValidationError) as exc_info:
        validate_records(records, "preference")
    assert any(issue.code == "IDENTICAL_PREFERENCE_PAIR" for issue in exc_info.value.issues)


def test_eval_missing_category_is_blocking() -> None:
    records = [
        {
            "id": "eval",
            "category": "",
            "prompt": "Answer",
            "expected_behavior": "Answer safely.",
            "constraints": {},
            "risk_tags": [],
        }
    ]
    with pytest.raises(DataValidationError) as exc_info:
        validate_records(records, "eval")
    assert any(issue.code == "MISSING_CATEGORY" for issue in exc_info.value.issues)
