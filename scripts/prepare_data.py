"""Validate sample/local datasets and write a small manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_reliability_lab.data import detect_data_issues, load_jsonl_dataset
from llm_reliability_lab.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/samples")
    parser.add_argument("--output-dir", default="outputs/sample_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)
    datasets = {
        "sft": input_dir / "sft_sample.jsonl",
        "preference": input_dir / "preference_sample.jsonl",
        "eval": input_dir / "eval_prompts_sample.jsonl",
    }

    manifest = {"datasets": {}, "notes": ["Sample data is synthetic and not a benchmark."]}
    for kind, path in datasets.items():
        records = load_jsonl_dataset(path, kind)  # type: ignore[arg-type]
        issues = detect_data_issues(records, kind)  # type: ignore[arg-type]
        manifest["datasets"][kind] = {
            "path": str(path),
            "num_records": len(records),
            "issues": [issue.__dict__ for issue in issues],
        }

    output_path = write_json(manifest, output_dir / "manifest.json")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
