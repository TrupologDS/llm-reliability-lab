from __future__ import annotations

from llm_reliability_lab.regression import detect_regressions


def _row(item_id: str, category: str, *, format_ok: bool, refusal: bool, score: float) -> dict:
    return {
        "id": item_id,
        "category": category,
        "response_word_count": 10,
        "is_refusal": refusal,
        "expected_refusal": False,
        "format_compliance": format_ok,
        "must_include_coverage": 1.0,
        "must_not_include_violation": False,
        "keyword_coverage": 1.0,
        "category_score": score,
    }


def test_detects_format_compliance_drop() -> None:
    baseline = [_row("a", "format_compliance", format_ok=True, refusal=False, score=1.0)]
    candidate = [_row("a", "format_compliance", format_ok=False, refusal=False, score=0.5)]
    regressions = detect_regressions(baseline, candidate, thresholds={"format_compliance_rate": 0.1})
    assert any(row["metric"] == "format_compliance_rate" and row["regression"] for row in regressions)


def test_detects_safe_prompt_refusal_increase() -> None:
    baseline = [_row("a", "instruction_following", format_ok=True, refusal=False, score=1.0)]
    candidate = [_row("a", "instruction_following", format_ok=True, refusal=True, score=0.7)]
    regressions = detect_regressions(baseline, candidate, thresholds={"refusal_rate": 0.1})
    assert any(row["metric"] == "refusal_rate_increase_on_safe_prompts" and row["regression"] for row in regressions)
