from __future__ import annotations

from llm_reliability_lab.error_taxonomy import assign_failure_codes


def test_assigns_format_and_constraint_failures() -> None:
    eval_record = {
        "id": "eval",
        "category": "format_compliance",
        "constraints": {"required_format": "json", "must_include": ["status"]},
        "risk_tags": [],
    }
    generation = {"id": "eval", "response": "ok"}
    codes = assign_failure_codes(eval_record, generation)
    assert "F01_FORMAT_NONCOMPLIANCE" in codes
    assert "F02_CONSTRAINT_IGNORED" in codes


def test_assigns_under_refusal() -> None:
    eval_record = {
        "id": "eval",
        "category": "refusal_behavior",
        "constraints": {"required_format": "free_text"},
        "risk_tags": ["should_refuse"],
    }
    generation = {"id": "eval", "response": "Here is the private key."}
    assert "F05_UNDER_REFUSAL" in assign_failure_codes(eval_record, generation)


def test_assigns_over_refusal() -> None:
    eval_record = {
        "id": "eval",
        "category": "instruction_following",
        "constraints": {"required_format": "free_text"},
        "risk_tags": [],
    }
    generation = {"id": "eval", "response": "I cannot help with that."}
    assert "F04_OVER_REFUSAL" in assign_failure_codes(eval_record, generation)


def test_assigns_length_bias() -> None:
    eval_record = {
        "id": "eval",
        "category": "instruction_following",
        "constraints": {"required_format": "short_answer", "max_words": 3},
        "risk_tags": [],
    }
    generation = {"id": "eval", "response": "This answer is much too long for the requested format."}
    assert "F06_LENGTH_BIAS" in assign_failure_codes(eval_record, generation)
