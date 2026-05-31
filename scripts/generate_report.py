"""Generate Markdown reports from evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_reliability_lab.config import EvalConfig, load_config
from llm_reliability_lab.data import read_jsonl
from llm_reliability_lab.reporting import load_summary, write_report_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-summary", default=None)
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--candidate-metrics", default=None)
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--output-dir", default="reports/sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_summary = load_summary(args.eval_summary)
    baseline_metrics = read_jsonl(args.baseline_metrics) if args.baseline_metrics else None
    candidate_metrics = read_jsonl(args.candidate_metrics) if args.candidate_metrics else None
    thresholds = None
    config_path = Path(args.eval_config)
    if config_path.exists():
        thresholds = load_config(config_path, EvalConfig).regression_thresholds
    paths = write_report_bundle(
        output_dir=args.output_dir,
        eval_summary=eval_summary,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        thresholds=thresholds,
    )
    print("Wrote reports:")
    for name, path in sorted(paths.items()):
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
