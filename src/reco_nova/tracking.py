"""MLflow experiment tracking utilities.

The module imports MLflow lazily so local model development and unit tests do
not require a configured tracking server. Databricks credentials must be
provided through environment variables, never source code or CLI arguments.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


def _numeric_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Flatten per-model report metrics into MLflow-compatible names."""
    output: dict[str, float] = {}
    for model_name, values in report["metrics"].items():
        for metric_name, value in values.items():
            if metric_name not in {"k", "users_evaluated"}:
                output[f"{model_name}.{metric_name}"] = float(value)
    for weight_name, values in report.get("hybrid_weight_tuning", {}).items():
        for metric_name, value in values.items():
            if metric_name not in {"k", "users_evaluated"}:
                output[f"tuning.{weight_name}.{metric_name}"] = float(value)
    return output


def log_recommender_run(
    report: Mapping[str, Any],
    artifacts_dir: Path,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None = None,
    task: str = "recommendation-evaluation",
    artifact_path: str = "recommender",
) -> dict[str, str]:
    """Log configuration, metrics, counts, and artifacts to MLflow."""
    try:
        import mlflow
    except Exception as exc:
        raise RuntimeError(
            "MLflow tracking was requested, but MLflow could not be imported. "
            "The environment may contain the obsolete conda-forge MLflow 1.x "
            "package. Reinstall tracking dependencies from environment.yml. "
            f"Original import error: {exc}"
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(report["configuration"])
        mlflow.log_params(
            {f"data.{name}": value for name, value in report["data"].items()}
        )
        mlflow.log_metrics(_numeric_metrics(report))
        mlflow.set_tags(
            {
                "project": "reco-nova",
                "task": task,
                "evaluation_scope": "warm-start",
            }
        )
        mlflow.log_artifacts(str(artifacts_dir), artifact_path=artifact_path)
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
        }


def log_baseline_run(**kwargs: Any) -> dict[str, str]:
    """Backward-compatible baseline tracking wrapper."""
    return log_recommender_run(
        **kwargs,
        task="baseline-collaborative-filtering",
        artifact_path="baseline",
    )


def log_policy_impact_run(
    report: Mapping[str, Any],
    artifacts_dir: Path,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None = None,
) -> dict[str, str]:
    """Log policy-impact simulation outputs to an MLflow server."""
    try:
        import mlflow
    except Exception as exc:
        raise RuntimeError(
            "MLflow tracking was requested, but MLflow could not be imported. "
            "Reinstall the tracking dependencies from environment.yml. "
            f"Original import error: {exc}"
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(report["configuration"])
        mlflow.log_params(
            {f"data.{name}": value for name, value in report["data"].items()}
        )

        metrics: dict[str, float] = {}
        for policy_name, values in report["offline_metrics"].items():
            for metric_name, value in values.items():
                if metric_name not in {"k", "users_evaluated"}:
                    numeric = float(value)
                    if math.isfinite(numeric):
                        metrics[
                            f"offline.{policy_name}.{metric_name}"
                        ] = numeric
        for policy_name, values in report["simulated_outcomes"].items():
            for metric_name, value in values.items():
                numeric = float(value)
                if math.isfinite(numeric):
                    metrics[f"simulated.{policy_name}.{metric_name}"] = numeric
        for metric_name, value in report["lift"].items():
            numeric = float(value)
            if math.isfinite(numeric):
                metrics[f"lift.{metric_name}"] = numeric
        for weight_name, values in report.get(
            "hybrid_weight_tuning", {}
        ).items():
            for metric_name, value in values.items():
                if metric_name not in {"k", "users_evaluated"}:
                    numeric = float(value)
                    if math.isfinite(numeric):
                        metrics[
                            f"tuning.{weight_name}.{metric_name}"
                        ] = numeric
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "project": "reco-nova",
                "task": "policy-impact-simulation",
                "evaluation_scope": "offline-simulation",
            }
        )
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="policy-impact")
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
        }
