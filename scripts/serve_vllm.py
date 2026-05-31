"""Print or start an optional vLLM server command."""

from __future__ import annotations

import argparse
import subprocess

from llm_reliability_lab.config import ServingConfig, load_config
from llm_reliability_lab.serving import build_vllm_command, render_vllm_instructions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--start", action="store_true", help="Start the vLLM server instead of printing help.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, ServingConfig)
    if args.start:
        subprocess.run(build_vllm_command(config), check=True)
    else:
        print(render_vllm_instructions(config))


if __name__ == "__main__":
    main()
