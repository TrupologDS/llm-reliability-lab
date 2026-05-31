"""Validate an SFT JSONL dataset before a real GPU run."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from llm_reliability_lab.data import (
    DataIssue,
    DataValidationError,
    detect_data_issues,
    load_jsonl_dataset,
)
from llm_reliability_lab.prompts import build_instruction_user_content
from llm_reliability_lab.reporting import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-suite", default="outputs/eval_suites/real_eval_suite.jsonl")
    parser.add_argument(
        "--report",
        default="reports/real_runs/qwen2_5_0_5b_lora_sft_run_001/data_validation_report.md",
    )
    parser.add_argument("--max-text-chars", type=int, default=16_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    eval_suite_path = Path(args.eval_suite)
    issues: list[DataIssue] = []
    records: list[dict[str, Any]] = []

    if "eval_suites" in dataset_path.parts:
        issues.append(
            DataIssue(
                code="EVAL_SUITE_AS_TRAINING_DATA",
                message="SFT dataset path points inside eval_suites/.",
                severity="error",
            )
        )

    try:
        records = load_jsonl_dataset(dataset_path, "sft")
        issues.extend(detect_data_issues(records, "sft", max_text_chars=args.max_text_chars))
    except DataValidationError as exc:
        issues.extend(exc.issues)

    if eval_suite_path.exists() and records:
        eval_records = load_jsonl_dataset(eval_suite_path, "eval")
        issues.extend(find_prompt_overlap(records, eval_records))
    elif not eval_suite_path.exists():
        issues.append(
            DataIssue(
                code="MISSING_EVAL_SUITE",
                message=f"Eval suite does not exist: {eval_suite_path}",
                severity="error",
            )
        )

    dataset_hash = sha256_file(dataset_path) if dataset_path.exists() else "TODO: dataset file missing"
    report = render_report(
        dataset_path=dataset_path,
        eval_suite_path=eval_suite_path,
        dataset_hash=dataset_hash,
        num_records=len(records),
        issues=issues,
    )
    write_report(args.report, report)

    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        print(f"SFT dataset validation failed with {len(errors)} error(s). Report: {args.report}")
        raise SystemExit(1)
    print(f"SFT dataset validation passed. Report: {args.report}")


def find_prompt_overlap(
    sft_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> list[DataIssue]:
    """Find exact normalized prompt overlap between SFT records and eval prompts."""

    eval_prompts = {normalize_text(record["prompt"]): record["id"] for record in eval_records}
    issues: list[DataIssue] = []
    for record in sft_records:
        prompt = normalize_text(build_instruction_user_content(record))
        if prompt in eval_prompts:
            issues.append(
                DataIssue(
                    code="TRAIN_EVAL_PROMPT_OVERLAP",
                    message=(f"SFT record {record['id']} exactly overlaps eval prompt {eval_prompts[prompt]}."),
                    severity="error",
                    record_id=record["id"],
                )
            )
    return issues


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_report(
    *,
    dataset_path: Path,
    eval_suite_path: Path,
    dataset_hash: str,
    num_records: int,
    issues: list[DataIssue],
) -> str:
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    return f"""# SFT Dataset Validation Report

## Summary

- Dataset: `{dataset_path.as_posix()}`
- Eval suite: `{eval_suite_path.as_posix()}`
- SHA256: `{dataset_hash}`
- Number of records: {num_records}
- Errors: {len(errors)}
- Warnings: {len(warnings)}

## Errors

{render_issues(errors)}

## Warnings

{render_issues(warnings)}

## Notes

- Evaluation prompts must not be used as training data.
- This report validates structure and overlap only; it does not judge data quality.
"""


def render_issues(issues: list[DataIssue]) -> str:
    if not issues:
        return "None."
    lines = []
    for issue in issues:
        location = f" record_id={issue.record_id}" if issue.record_id else ""
        lines.append(f"- `{issue.code}`{location}: {issue.message}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
