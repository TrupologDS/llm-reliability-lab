"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_reliability_lab.regression import detect_regressions
from llm_reliability_lab.utils import ensure_dir, read_json

TODO_NOTE = "TODO: Fill this section after running a real experiment."


def write_report(path: str | Path, content: str) -> Path:
    """Write one Markdown report."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return output_path


def write_report_bundle(
    *,
    output_dir: str | Path = "reports",
    eval_summary: dict[str, Any] | None = None,
    baseline_metrics: list[dict[str, Any]] | None = None,
    candidate_metrics: list[dict[str, Any]] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Path]:
    """Generate the standard report bundle."""

    report_dir = ensure_dir(output_dir)
    regressions = []
    if baseline_metrics is not None and candidate_metrics is not None:
        regressions = detect_regressions(
            baseline_metrics,
            candidate_metrics,
            thresholds=thresholds,
            include_non_regressions=True,
        )

    paths = {
        "eval_report": write_report(report_dir / "eval_report.md", render_eval_report(eval_summary)),
        "regression_report": write_report(
            report_dir / "regression_report.md",
            render_regression_report(regressions),
        ),
        "error_analysis": write_report(
            report_dir / "error_analysis.md",
            render_error_analysis(eval_summary),
        ),
        "model_card": write_report(report_dir / "model_card.md", render_model_card(eval_summary)),
        "data_card": write_report(report_dir / "data_card.md", render_data_card(eval_summary)),
        "reproducibility_checklist": write_report(
            report_dir / "reproducibility_checklist.md",
            render_reproducibility_checklist(eval_summary),
        ),
    }
    return paths


def load_summary(path: str | Path | None) -> dict[str, Any] | None:
    """Load an evaluation summary JSON if provided."""

    if path is None:
        return None
    return read_json(path)


def render_eval_report(summary: dict[str, Any] | None) -> str:
    """Render the evaluation report."""

    summary = summary or {}
    aggregates = summary.get("aggregates") or []
    notes = summary.get("notes") or [TODO_NOTE]
    mock_note = ""
    if summary.get("backend") == "mock":
        mock_note = (
            "\nThese metrics come from the deterministic mock backend. "
            "They validate the pipeline and are not benchmark results.\n"
        )
    metrics_table = _markdown_table(
        aggregates,
        [
            "category",
            "num_examples",
            "format_compliance_rate",
            "must_include_coverage",
            "must_not_include_violation_rate",
            "refusal_rate",
            "category_score",
        ],
    )
    return f"""# Evaluation Report

## Experiment Summary

- Model: {summary.get("model", "TODO")}
- Backend: {summary.get("backend", "TODO")}
- Evaluation suite: {summary.get("eval_prompts_path", "TODO")}
- Number of prompts: {summary.get("num_prompts", "TODO")}
- Status: {summary.get("status", "TODO")}

## Metrics

{mock_note}
{metrics_table or TODO_NOTE}

## Category-Level Results

{metrics_table or TODO_NOTE}

## Limitations

{_bullet_list(notes)}

TODO: Add interpretation after real baseline and candidate evaluations are complete.
"""


def render_regression_report(regressions: list[dict[str, Any]] | None = None) -> str:
    """Render baseline vs candidate regression report."""

    regressions = regressions or []
    table = _markdown_table(
        regressions,
        ["category", "metric", "baseline", "candidate", "delta", "threshold", "regression", "severity"],
    )
    return f"""# Regression Report

## Baseline vs Candidate Comparison

{table or TODO_NOTE}

## Regressions By Category

TODO: Add selected examples for each regression after running real baseline and candidate evaluations.

## Notes

- Regression detection uses rule-based metrics and configured thresholds.
- A flagged regression is a review target, not an automatic model rejection.
"""


def render_error_analysis(summary: dict[str, Any] | None) -> str:
    """Render qualitative error analysis report."""

    summary = summary or {}
    failure_table = _markdown_table(summary.get("failure_summary") or [], ["failure_code", "count", "description"])
    return f"""# Error Analysis

## Failure Taxonomy

{failure_table or TODO_NOTE}

## Top Failure Categories

TODO: Review generated outputs and fill qualitative observations.

## Selected Examples

TODO: Add representative examples with prompt, response, expected behavior, and assigned failure codes.

## Debugging Hypotheses

TODO: Link failures to data, prompt formatting, decoding config, or training changes.

## Next Experiments

TODO: Propose targeted eval additions or training/data changes.
"""


def render_model_card(summary: dict[str, Any] | None) -> str:
    """Render model card template."""

    summary = summary or {}
    return f"""# Model Card

## Model Description

- Base or candidate model: {summary.get("model", "TODO")}
- Adapter/checkpoint path: TODO
- Training method: TODO, for example SFT with LoRA or DPO-style preference optimization

## Intended Use

TODO: Describe intended research and evaluation use.

## Not Intended Use

TODO: Describe out-of-scope or unsafe deployment uses.

## Training Data Summary

TODO: Document data sources, licenses, filtering, and known limitations.

## Evaluation Summary

TODO: Add real evaluation results. Do not use mock sample outputs as benchmark evidence.

## Limitations

- Small models may fail on complex reasoning and safety-sensitive tasks.
- Rule-based metrics do not prove semantic correctness.
- TODO: Add limitations observed in real runs.

## Ethical and Safety Considerations

TODO: Document misuse risks, privacy considerations, and refusal behavior findings.

## Reproducibility Information

TODO: Add model revision, dataset revision, configs, seeds, hardware, package versions, and commands.
"""


def render_data_card(summary: dict[str, Any] | None) -> str:
    """Render data card template."""

    summary = summary or {}
    return f"""# Data Card

## Data Sources

- Evaluation suite: {summary.get("eval_prompts_path", "TODO")}
- SFT data: TODO
- Preference data: TODO

## Schema

- SFT: id, system, instruction, input, response, source
- Preference: id, prompt, chosen, rejected, source
- Evaluation: id, category, prompt, expected_behavior, constraints, risk_tags

## Preprocessing

TODO: Document normalization, filtering, deduplication, and splits.

## Filtering

TODO: Document removed records and reasons.

## Known Limitations

- Sample data is synthetic and tiny.
- TODO: Add real dataset coverage gaps.

## Privacy and Licensing Notes

TODO: Confirm dataset licenses and absence of private or sensitive data.
"""


def render_reproducibility_checklist(summary: dict[str, Any] | None) -> str:
    """Render reproducibility checklist."""

    summary = summary or {}
    return f"""# Reproducibility Checklist

- [ ] Seeds recorded
- [ ] Config files saved
- [ ] Package versions exported
- [ ] Model versions and revisions recorded
- [ ] Dataset versions and revisions recorded
- [ ] Hardware notes recorded
- [ ] Commands to reproduce included
- [ ] MLflow run ID linked
- [ ] Reports generated from real outputs

## Current Run Metadata

- Model: {summary.get("model", "TODO")}
- Backend: {summary.get("backend", "TODO")}
- Evaluation prompts: {summary.get("eval_prompts_path", "TODO")}

## Commands

TODO: Add exact commands used for real SFT, DPO, evaluation, and reporting.
"""


def render_training_report(
    *,
    title: str,
    config: dict[str, Any],
    output_dir: str,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Render a compact training summary."""

    metrics = metrics or {}
    return f"""# {title}

## Status

TODO: Replace this placeholder with real training observations after the run completes.

## Output Directory

`{output_dir}`

## Config Snapshot

```json
{_json_block(config)}
```

## Metrics

```json
{_json_block(metrics)}
```
"""


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_format_cell(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ")


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _json_block(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, ensure_ascii=True, sort_keys=True)
