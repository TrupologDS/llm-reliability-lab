"""Evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_reliability_lab.config import EvalConfig
from llm_reliability_lab.data import load_jsonl_dataset, write_jsonl
from llm_reliability_lab.error_taxonomy import assign_failures_for_rows, summarize_failures
from llm_reliability_lab.inference import InferenceBackend, generate_for_eval_records
from llm_reliability_lab.metrics import aggregate_metrics, evaluate_generation
from llm_reliability_lab.utils import ensure_dir, write_json


def run_evaluation(
    config: EvalConfig,
    *,
    backend: InferenceBackend = "mock",
    model_name_or_path: str | None = None,
    output_dir: str | Path = "outputs/eval",
) -> dict[str, Any]:
    """Run generation, metrics, and failure assignment for one model."""

    eval_records = load_jsonl_dataset(config.eval_prompts_path, "eval")
    model_name = model_name_or_path or config.candidate_model
    output_path = ensure_dir(output_dir)

    generations = generate_for_eval_records(
        eval_records,
        model_name_or_path=model_name,
        backend=backend,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        template_backend=config.template_backend,
        adapter_base_model_name_or_path=config.candidate_base_model,
    )
    metrics = evaluate_generations(eval_records, generations)
    eval_by_id = {record["id"]: record for record in eval_records}
    metrics_by_id = {row["id"]: row for row in metrics}
    failure_assignments = assign_failures_for_rows(eval_by_id, generations, metrics_by_id)

    generation_path = output_path / "generations.jsonl"
    metrics_path = output_path / "metrics.jsonl"
    failures_path = output_path / "failure_assignments.jsonl"
    write_jsonl(generations, generation_path)
    write_jsonl(metrics, metrics_path)
    write_jsonl(failure_assignments, failures_path)

    status = "sample_pipeline_check" if backend == "mock" else "model_evaluation_output"
    summary = {
        "status": status,
        "model": model_name,
        "backend": backend,
        "num_prompts": len(eval_records),
        "eval_prompts_path": config.eval_prompts_path,
        "generation_path": str(generation_path),
        "metrics_path": str(metrics_path),
        "failure_assignments_path": str(failures_path),
        "aggregates": aggregate_metrics(metrics),
        "failure_summary": summarize_failures(failure_assignments),
        "notes": [
            "TODO: Replace mock outputs with real model generations before reporting results.",
            "No LLM-as-judge scores are computed by default.",
        ],
    }
    write_json(summary, output_path / "eval_summary.json")
    return summary


def evaluate_generations(
    eval_records: list[dict[str, Any]],
    generations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate already generated responses."""

    eval_by_id = {record["id"]: record for record in eval_records}
    rows: list[dict[str, Any]] = []
    for generation in generations:
        item_id = generation.get("id")
        if item_id not in eval_by_id:
            msg = f"Generation id '{item_id}' is not present in eval records."
            raise KeyError(msg)
        rows.append(evaluate_generation(generation, eval_by_id[item_id]))
    return rows
