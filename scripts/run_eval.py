"""Run evaluation generation and metrics."""

from __future__ import annotations

import argparse

from llm_reliability_lab.config import EvalConfig, load_config
from llm_reliability_lab.evaluate import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--backend", choices=["mock", "hf", "vllm"], default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default="outputs/eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, EvalConfig)
    summary = run_evaluation(
        config,
        backend=args.backend,  # type: ignore[arg-type]
        model_name_or_path=args.model,
        output_dir=args.output_dir,
    )
    print(f"Wrote evaluation summary for {summary['num_prompts']} prompts to {args.output_dir}")


if __name__ == "__main__":
    main()
