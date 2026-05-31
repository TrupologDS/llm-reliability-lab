"""CLI for DPO-style preference optimization."""

from __future__ import annotations

import argparse

from llm_reliability_lab.config import DPOConfig, MLflowConfig, ModelConfig, load_config
from llm_reliability_lab.dpo import train_dpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--dpo-config", default="configs/dpo.yaml")
    parser.add_argument("--mlflow-config", default="configs/mlflow.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_dpo(
        model_config=load_config(args.model_config, ModelConfig),
        dpo_config=load_config(args.dpo_config, DPOConfig),
        mlflow_config=load_config(args.mlflow_config, MLflowConfig),
    )
    print(summary)


if __name__ == "__main__":
    main()
