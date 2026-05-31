"""Thin MLflow helpers with a no-op fallback."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_reliability_lab.config import MLflowConfig


class MLflowRun:
    """Context manager that logs to MLflow only when enabled and installed."""

    def __init__(self, config: MLflowConfig, *, run_name: str | None = None) -> None:
        self.config = config
        self.run_name = run_name or config.run_name
        self._mlflow: Any | None = None
        self.enabled = config.enabled

    def __enter__(self) -> MLflowRun:
        if not self.enabled:
            return self
        try:
            import mlflow
        except ImportError:
            self.enabled = False
            return self

        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.end_run()

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log scalar-ish params."""

        if not self.enabled or self._mlflow is None:
            return
        sanitized = {key: _stringify(value) for key, value in params.items()}
        self._mlflow.log_params(sanitized)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log numeric metrics."""

        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: str | Path) -> None:
        """Log one artifact if configured."""

        if not self.config.log_artifacts or not self.enabled or self._mlflow is None:
            return
        self._mlflow.log_artifact(str(path))

    def log_artifacts(self, path: str | Path) -> None:
        """Log an artifact directory if configured."""

        if not self.config.log_artifacts or not self.enabled or self._mlflow is None:
            return
        self._mlflow.log_artifacts(str(path))

    def log_trace_placeholder(self, name: str, payload: Mapping[str, Any]) -> None:
        """Placeholder for optional GenAI tracing."""

        if not self.config.log_traces:
            return
        _ = (name, payload)
        # TODO: Add MLflow GenAI tracing once the desired schema is finalized.


def _stringify(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)
