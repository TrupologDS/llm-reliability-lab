"""Build a merged evaluation suite from JSONL suite files."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_reliability_lab.data import detect_data_issues, load_jsonl_dataset, write_jsonl
from llm_reliability_lab.utils import ensure_dir, write_json

DEFAULT_SUITES = [
    "instruction_following",
    "format_compliance",
    "multilingual",
    "refusal_behavior",
    "grounded_qa",
    "robustness_perturbations",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", default="eval_suites")
    parser.add_argument("--output", default="outputs/eval_suites/real_eval_suite.jsonl")
    parser.add_argument("--manifest", default="outputs/eval_suites/real_eval_suite_manifest.json")
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    merged: list[dict] = []
    manifest = {
        "output": str(output_path),
        "suites": [],
        "notes": [
            "Merged suite contains synthetic evaluation prompts only.",
            "Do not use these prompts as SFT or preference training data.",
        ],
    }
    seen_ids: set[str] = set()
    for suite_name in args.suites:
        suite_path = suite_dir / f"{suite_name}.jsonl"
        records = load_jsonl_dataset(suite_path, "eval")
        duplicates = sorted({record["id"] for record in records if record["id"] in seen_ids})
        if duplicates:
            duplicate_list = ", ".join(duplicates[:10])
            raise ValueError(f"Duplicate eval ids across suites: {duplicate_list}")
        seen_ids.update(record["id"] for record in records)
        issues = detect_data_issues(records, "eval")
        merged.extend(records)
        manifest["suites"].append(
            {
                "name": suite_name,
                "path": str(suite_path),
                "num_records": len(records),
                "warnings": [issue.__dict__ for issue in issues if issue.severity == "warning"],
            }
        )

    write_jsonl(merged, output_path)
    manifest["num_records"] = len(merged)
    manifest["categories"] = sorted({record["category"] for record in merged})
    write_json(manifest, args.manifest)
    print(f"Wrote {len(merged)} prompts to {output_path}")


if __name__ == "__main__":
    main()
