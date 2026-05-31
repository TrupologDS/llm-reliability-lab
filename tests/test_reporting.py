from __future__ import annotations

from llm_reliability_lab.reporting import render_eval_report, write_report_bundle


def test_eval_report_contains_todo_with_missing_summary() -> None:
    report = render_eval_report(None)
    assert "TODO" in report
    assert "Evaluation Report" in report


def test_write_report_bundle(tmp_path) -> None:
    summary = {
        "model": "mock",
        "backend": "mock",
        "num_prompts": 1,
        "eval_prompts_path": "sample.jsonl",
        "aggregates": [
            {
                "category": "overall",
                "num_examples": 1,
                "format_compliance_rate": 1.0,
                "must_include_coverage": 1.0,
                "must_not_include_violation_rate": 0.0,
                "refusal_rate": 0.0,
                "category_score": 1.0,
            }
        ],
    }
    paths = write_report_bundle(output_dir=tmp_path, eval_summary=summary)
    assert paths["eval_report"].exists()
    assert paths["model_card"].exists()
    assert "TODO" in paths["model_card"].read_text(encoding="utf-8")
