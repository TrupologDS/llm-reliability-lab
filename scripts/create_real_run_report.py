"""Create a real-run report directory without inventing results."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml

from llm_reliability_lab.reporting import write_report
from llm_reliability_lab.utils import ensure_dir, read_json

DEFAULT_RUN_NAME = "qwen2_5_0_5b_lora_sft_run_001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=f"reports/real_runs/{DEFAULT_RUN_NAME}")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--sft-config", default="configs/sft_real.yaml")
    parser.add_argument("--eval-config", default="configs/eval_real.yaml")
    parser.add_argument("--mlflow-config", default="configs/mlflow.yaml")
    parser.add_argument("--comparison-dir", default=f"outputs/{DEFAULT_RUN_NAME}/comparison")
    parser.add_argument("--baseline-eval-dir", default=f"outputs/{DEFAULT_RUN_NAME}/baseline_eval")
    parser.add_argument("--sft-output-dir", default=f"outputs/{DEFAULT_RUN_NAME}/sft")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = ensure_dir(args.run_dir)
    paths = RunPaths(
        run_dir=run_dir,
        comparison_dir=Path(args.comparison_dir),
        baseline_eval_dir=Path(args.baseline_eval_dir),
        sft_output_dir=Path(args.sft_output_dir),
    )

    snapshot = build_config_snapshot(
        {
            "model": Path(args.model_config),
            "sft": Path(args.sft_config),
            "eval": Path(args.eval_config),
            "mlflow": Path(args.mlflow_config),
        }
    )
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    write_report(run_dir / "README.md", render_run_readme(paths, snapshot))
    write_or_todo(
        source=paths.comparison_dir / "regression_report.md",
        target=run_dir / "regression_report.md",
        fallback=render_missing_regression_report(),
    )
    write_or_todo(
        source=paths.comparison_dir / "qualitative_examples.md",
        target=run_dir / "qualitative_examples.md",
        fallback=render_missing_qualitative_examples(),
    )
    write_report(run_dir / "eval_report.md", render_eval_report(paths))
    write_report(run_dir / "error_analysis.md", render_error_analysis(paths))
    write_report(run_dir / "model_card.md", render_model_card(snapshot))
    write_report(run_dir / "data_card.md", render_data_card(snapshot))
    print(f"Wrote real-run report scaffold to {run_dir}")


class RunPaths:
    """Filesystem locations for a real run."""

    def __init__(
        self,
        *,
        run_dir: Path,
        comparison_dir: Path,
        baseline_eval_dir: Path,
        sft_output_dir: Path,
    ) -> None:
        self.run_dir = run_dir
        self.comparison_dir = comparison_dir
        self.baseline_eval_dir = baseline_eval_dir
        self.sft_output_dir = sft_output_dir


def build_config_snapshot(config_paths: dict[str, Path]) -> dict[str, Any]:
    """Read YAML configs into one immutable-ish snapshot file."""

    snapshot: dict[str, Any] = {}
    for name, path in config_paths.items():
        if path.exists():
            snapshot[name] = {
                "path": path.as_posix(),
                "content": yaml.safe_load(path.read_text(encoding="utf-8")) or {},
            }
        else:
            snapshot[name] = {"path": path.as_posix(), "content": "TODO: config file missing"}
    return snapshot


def write_or_todo(*, source: Path, target: Path, fallback: str) -> None:
    """Copy a generated report if present; otherwise write a TODO placeholder."""

    if source.exists():
        ensure_dir(target.parent)
        shutil.copyfile(source, target)
    else:
        write_report(target, fallback)


def render_run_readme(paths: RunPaths, snapshot: dict[str, Any]) -> str:
    model_name = snapshot.get("model", {}).get("content", {}).get("model_name_or_path", "TODO")
    candidate = snapshot.get("eval", {}).get("content", {}).get("candidate_model", "TODO")
    return f"""# Qwen2.5 0.5B LoRA SFT Run 001

## Status

TODO: Complete the GPU run and replace this status with observed run notes.

## Experiment

- Baseline model: `{model_name}`
- Candidate adapter/model path: `{candidate}`
- SFT output directory: `{to_posix(paths.sft_output_dir)}`
- Baseline evaluation directory: `{to_posix(paths.baseline_eval_dir)}`
- Comparison directory: `{to_posix(paths.comparison_dir)}`

## Commands

```bash
python -m pip install -e ".[train,eval,dev]"
make build-real-eval-suite
make validate-sft-real SFT_REAL_DATASET=data/processed/qwen2_5_0_5b_sft_train.jsonl
make preflight-sft-format SFT_REAL_DATASET=data/processed/qwen2_5_0_5b_sft_train.jsonl
make eval-baseline
make train-sft-real
make compare-baseline-sft
make real-run-report
```

Optional merged-model workflow:

```bash
make merge-sft-lora
make compare-baseline-sft-merged
```

## Required Before Claiming Results

- TODO: Record GPU model/count, CUDA version, driver version, CPU, RAM, and OS.
- TODO: Record exact model revisions and dataset file hashes.
- TODO: Review qualitative examples and failure taxonomy.
- TODO: Document limitations and next experiments.

No improvement is claimed until real outputs are generated and reviewed.
"""


def render_eval_report(paths: RunPaths) -> str:
    baseline_summary = read_optional_json(paths.baseline_eval_dir / "eval_summary.json")
    comparison_summary = read_optional_json(paths.comparison_dir / "comparison_summary.json")
    return f"""# Evaluation Report

## Baseline Evaluation

{render_summary_block(baseline_summary, "baseline evaluation")}

## Candidate Evaluation

{render_summary_block(comparison_summary.get("candidate") if comparison_summary else None, "candidate evaluation")}

## Notes

- TODO: Add interpretation after real model outputs are reviewed.
- Do not use sample pipeline outputs as evidence of model behavior.
"""


def render_missing_regression_report() -> str:
    return """# Regression Report

TODO: Run `make compare-baseline-sft` after SFT completes.

No regression results are available yet.
"""


def render_missing_qualitative_examples() -> str:
    return """# Qualitative Examples

TODO: Run `make compare-baseline-sft` and review selected examples.

No qualitative examples are available yet.
"""


def render_error_analysis(paths: RunPaths) -> str:
    failure_path = paths.comparison_dir / "candidate" / "failure_assignments.jsonl"
    status = (
        f"Candidate failure assignments found at `{failure_path}`."
        if failure_path.exists()
        else "TODO: Candidate failure assignments are missing until comparison is run."
    )
    return f"""# Error Analysis

## Failure Taxonomy

{status}

## Top Failure Categories

TODO: Summarize failure codes after reviewing real outputs.

## Debugging Hypotheses

TODO: Link failures to data, training config, decoding config, or base model limitations.

## Next Experiments

TODO: Define targeted follow-up changes.
"""


def render_model_card(snapshot: dict[str, Any]) -> str:
    model_name = snapshot.get("model", {}).get("content", {}).get("model_name_or_path", "TODO")
    candidate = snapshot.get("eval", {}).get("content", {}).get("candidate_model", "TODO")
    return f"""# Model Card

## Model Description

- Base model: `{model_name}`
- Candidate adapter/model path: `{candidate}`
- Training method: LoRA SFT

## Intended Use

TODO: Describe intended evaluation/research use after the run is complete.

## Evaluation Summary

TODO: Add real evaluation results. No results are claimed yet.

## Limitations

TODO: Add observed limitations from real outputs.

## Reproducibility

See `config_snapshot.yaml`.
TODO: Add model revision, package versions, hardware, and commands.
"""


def render_data_card(snapshot: dict[str, Any]) -> str:
    sft_path = snapshot.get("sft", {}).get("content", {}).get("dataset_path", "TODO")
    eval_path = snapshot.get("eval", {}).get("content", {}).get("eval_prompts_path", "TODO")
    return f"""# Data Card

## Data Sources

- SFT data: `{sft_path}`
- Evaluation suite: `{eval_path}`

## Notes

- Evaluation prompts are synthetic reliability prompts and are not training data.
- TODO: Record SFT data source, license, preprocessing, filtering, and file hash.
- TODO: Confirm no private or sensitive data is present.

## Limitations

TODO: Add coverage gaps after reviewing the run.
"""


def render_summary_block(summary: dict[str, Any] | None, name: str) -> str:
    if not summary:
        return f"TODO: {name} output is missing."
    return "\n".join(
        [
            f"- Model: `{summary.get('model', 'TODO')}`",
            f"- Backend: `{summary.get('backend', 'TODO')}`",
            f"- Status: `{summary.get('status', 'TODO')}`",
            f"- Number of prompts: `{summary.get('num_prompts', 'TODO')}`",
        ]
    )


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def to_posix(path: Path) -> str:
    """Render paths consistently in Markdown reports."""

    return path.as_posix()


if __name__ == "__main__":
    main()
