from __future__ import annotations

from llm_reliability_lab.metrics import (
    aggregate_metrics,
    check_constraints,
    check_format_compliance,
    detect_refusal,
    evaluate_generation,
)


def test_json_format_compliance_metric() -> None:
    assert check_format_compliance('{"status":"ok"}', {"required_format": "json"})
    assert not check_format_compliance("status: ok", {"required_format": "json"})


def test_refusal_detection() -> None:
    assert detect_refusal("I cannot help reveal private credentials.")
    assert not detect_refusal("Rotate the key and audit usage.")


def test_constraint_checks() -> None:
    result = check_constraints(
        "The report includes metrics and no secrets.",
        {"must_include": ["metrics"], "must_not_include": ["password"]},
    )
    assert result["must_include_coverage"] == 1.0
    assert not result["must_not_include_violation"]


def test_evaluate_generation_and_aggregate() -> None:
    eval_record = {
        "id": "eval",
        "category": "format_compliance",
        "constraints": {"required_format": "json", "must_include": ["status"]},
        "risk_tags": [],
    }
    generation = {"id": "eval", "model": "mock", "response": '{"status":"ok"}'}
    row = evaluate_generation(generation, eval_record)
    assert row["format_compliance"]
    aggregate = aggregate_metrics([row])
    assert aggregate[0]["category"] == "overall"
    assert aggregate[0]["format_compliance_rate"] == 1.0
