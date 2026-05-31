"""Run baseline vs candidate evaluation and regression analysis."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from llm_reliability_lab.config import EvalConfig, load_config
from llm_reliability_lab.data import load_jsonl_dataset, read_jsonl, write_jsonl
from llm_reliability_lab.evaluate import run_evaluation
from llm_reliability_lab.regression import detect_regressions
from llm_reliability_lab.reporting import render_regression_report, write_report
from llm_reliability_lab.utils import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline model name/path.")
    parser.add_argument("--candidate", required=True, help="Candidate model name/path.")
    parser.add_argument("--backend", choices=["mock", "hf", "vllm"], default="hf")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--output-dir", default="outputs/model_comparison")
    parser.add_argument(
        "--reuse-baseline-dir",
        default=None,
        help="Existing baseline eval directory with generations.jsonl, metrics.jsonl, and eval_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, EvalConfig)
    output_dir = ensure_dir(args.output_dir)

    baseline_dir = output_dir / "baseline"
    candidate_dir = output_dir / "candidate"
    if args.reuse_baseline_dir:
        copy_reused_baseline(Path(args.reuse_baseline_dir), baseline_dir)
        baseline_summary = read_jsonl_summary(baseline_dir / "eval_summary.json")
    else:
        baseline_summary = run_evaluation(
            config,
            backend=args.backend,
            model_name_or_path=args.baseline,
            output_dir=baseline_dir,
        )
    candidate_summary = run_evaluation(
        config,
        backend=args.backend,
        model_name_or_path=args.candidate,
        output_dir=candidate_dir,
    )

    baseline_metrics = read_jsonl(baseline_dir / "metrics.jsonl")
    candidate_metrics = read_jsonl(candidate_dir / "metrics.jsonl")
    regressions = detect_regressions(
        baseline_metrics,
        candidate_metrics,
        thresholds=config.regression_thresholds,
        include_non_regressions=True,
    )

    baseline_generations = read_jsonl(baseline_dir / "generations.jsonl")
    candidate_generations = read_jsonl(candidate_dir / "generations.jsonl")
    eval_records = load_jsonl_dataset(config.eval_prompts_path, "eval")
    examples = select_qualitative_examples(
        eval_records=eval_records,
        baseline_generations=baseline_generations,
        candidate_generations=candidate_generations,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        regressions=regressions,
    )

    write_json({"regressions": regressions}, output_dir / "regressions.json")
    write_jsonl(examples, output_dir / "qualitative_examples.jsonl")
    write_report(output_dir / "regression_report.md", render_regression_report(regressions))
    write_report(output_dir / "qualitative_examples.md", render_qualitative_examples(examples))
    write_json(
        {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "regression_report": str(output_dir / "regression_report.md"),
            "qualitative_examples": str(output_dir / "qualitative_examples.jsonl"),
            "notes": [
                "Mock backend comparisons are pipeline checks only.",
                "Real model comparisons require identical eval suites and decoding configs.",
            ],
        },
        output_dir / "comparison_summary.json",
    )
    print(f"Wrote comparison outputs to {output_dir}")


def select_qualitative_examples(
    *,
    eval_records: list[dict[str, Any]],
    baseline_generations: list[dict[str, Any]],
    candidate_generations: list[dict[str, Any]],
    baseline_metrics: list[dict[str, Any]],
    candidate_metrics: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Select examples linked to detected regressions and largest score drops."""

    eval_by_id = {record["id"]: record for record in eval_records}
    baseline_by_id = {record["id"]: record for record in baseline_generations}
    candidate_by_id = {record["id"]: record for record in candidate_generations}
    baseline_metrics_by_id = {record["id"]: record for record in baseline_metrics}
    candidate_metrics_by_id = {record["id"]: record for record in candidate_metrics}
    regressed_categories = {
        row["category"] for row in regressions if row.get("regression") and row["category"] != "overall"
    }

    scored_ids: list[tuple[float, str]] = []
    for item_id, baseline_row in baseline_metrics_by_id.items():
        candidate_row = candidate_metrics_by_id.get(item_id)
        if candidate_row is None:
            continue
        score_drop = float(baseline_row.get("category_score", 0.0)) - float(candidate_row.get("category_score", 0.0))
        category = str(baseline_row.get("category", ""))
        category_bonus = 1.0 if category in regressed_categories else 0.0
        scored_ids.append((category_bonus + score_drop, item_id))

    selected: list[dict[str, Any]] = []
    for _, item_id in sorted(scored_ids, reverse=True)[:limit]:
        eval_record = eval_by_id[item_id]
        selected.append(
            {
                "id": item_id,
                "category": eval_record["category"],
                "prompt": eval_record["prompt"],
                "expected_behavior": eval_record["expected_behavior"],
                "baseline_response": baseline_by_id[item_id]["response"],
                "candidate_response": candidate_by_id[item_id]["response"],
                "baseline_metrics": baseline_metrics_by_id[item_id],
                "candidate_metrics": candidate_metrics_by_id[item_id],
            }
        )
    return selected


def render_qualitative_examples(examples: list[dict[str, Any]]) -> str:
    """Render qualitative comparison examples."""

    if not examples:
        return "# Qualitative Examples\n\nTODO: No examples selected yet.\n"
    sections = ["# Qualitative Examples"]
    for example in examples:
        sections.append(
            "\n".join(
                [
                    f"## {example['id']} ({example['category']})",
                    "",
                    f"Prompt: {example['prompt']}",
                    "",
                    f"Expected behavior: {example['expected_behavior']}",
                    "",
                    f"Baseline response: {example['baseline_response']}",
                    "",
                    f"Candidate response: {example['candidate_response']}",
                    "",
                    "Reviewer notes: TODO",
                ]
            )
        )
    return "\n\n".join(sections)


def read_jsonl_summary(path: Path) -> dict[str, Any]:
    """Read a required evaluation summary from a reused evaluation directory."""

    if not path.exists():
        msg = f"Missing baseline summary: {path}"
        raise FileNotFoundError(msg)
    return read_json(path)


def copy_reused_baseline(source_dir: Path, target_dir: Path) -> None:
    """Copy existing baseline artifacts into the comparison directory."""

    required_files = [
        "eval_summary.json",
        "generations.jsonl",
        "metrics.jsonl",
        "failure_assignments.jsonl",
    ]
    ensure_dir(target_dir)
    for filename in required_files:
        source = source_dir / filename
        if not source.exists():
            msg = f"Missing reused baseline artifact: {source}"
            raise FileNotFoundError(msg)
        shutil.copyfile(source, target_dir / filename)


if __name__ == "__main__":
    main()
