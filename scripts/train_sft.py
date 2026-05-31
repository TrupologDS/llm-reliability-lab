"""CLI for LoRA SFT."""

from __future__ import annotations

import argparse

from llm_reliability_lab.config import MLflowConfig, ModelConfig, SFTConfig, load_config
from llm_reliability_lab.sft import train_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--sft-config", default="configs/sft.yaml")
    parser.add_argument("--mlflow-config", default="configs/mlflow.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_sft(
        model_config=load_config(args.model_config, ModelConfig),
        sft_config=load_config(args.sft_config, SFTConfig),
        mlflow_config=load_config(args.mlflow_config, MLflowConfig),
    )
    print(summary)


if __name__ == "__main__":
    main()
