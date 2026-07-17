"""Final held-out test evaluation for all recommendation approaches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reco_nova.evaluation import (
    bootstrap_metric_intervals,
    catalog_coverage,
    evaluate_rankings,
    fresh_catalog_coverage_at_k,
    fresh_share_at_k,
    users_with_fresh_hit_at_k,
)
from reco_nova.models import (
    CollaborativeSVD,
    ContentRecommender,
    HybridRecommender,
    PopularityRecommender,
)
from reco_nova.tracking import log_recommender_run
from reco_nova.train import _positive_limit, build_ground_truth


def _recommend(model: object, users: list[str], k: int) -> dict[str, list[str]]:
    return {
        user: [item for item, _ in model.recommend(user, k)]  # type: ignore[attr-defined]
        for user in users
    }


def _evaluate_model(
    recommendations: dict[str, list[str]],
    truth: dict[str, set[str]],
    catalog_size: int,
    k: int,
    bootstrap_samples: int,
    random_state: int,
    fresh_item_ids: set[str] | None = None,
) -> dict[str, int | float]:
    metrics = evaluate_rankings(recommendations, truth, k).to_dict()
    metrics["catalog_coverage_at_k"] = catalog_coverage(
        recommendations, catalog_size, k
    )
    intervals = bootstrap_metric_intervals(
        recommendations,
        truth,
        k=k,
        samples=bootstrap_samples,
        random_state=random_state,
    )
    for name, (lower, upper) in intervals.items():
        metrics[f"{name}_ci95_lower"] = lower
        metrics[f"{name}_ci95_upper"] = upper
    if fresh_item_ids is not None:
        metrics["fresh_catalog_coverage_at_k"] = fresh_catalog_coverage_at_k(
            recommendations, fresh_item_ids, k
        )
        metrics["fresh_share_at_k"] = fresh_share_at_k(
            recommendations, fresh_item_ids, k
        )
        metrics["users_with_fresh_hit_at_k"] = users_with_fresh_hit_at_k(
            recommendations, fresh_item_ids, k
        )
    return metrics


def _markdown(report: dict[str, object]) -> str:
    config = report["configuration"]
    data = report["data"]
    metrics = report["metrics"]
    lines = [
        "# Reco-Nova Final Offline Evaluation",
        "",
        "This report evaluates frozen model choices once on the untouched test split. ",
        "Hybrid weighting was selected on validation data before this evaluation.",
        "",
        "## Evaluation setup",
        "",
        f"- Training interactions: {data['training_rows']:,}",
        f"- Eligible training users: {data['training_users']:,}",
        f"- Eligible catalog items: {data['training_items']:,}",
        f"- Warm test users evaluated: {data['warm_test_users']:,}",
        f"- Ranking cutoff: K={config['k']}",
        f"- Frozen collaborative weight: {config['collaborative_weight']}",
        f"- Bootstrap samples: {config['bootstrap_samples']}",
        "",
        "## Results",
        "",
        "| Model | NDCG@K | MAP@K | Hit Rate@K | Catalog Coverage@K |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['ndcg_at_k']:.6f} | "
            f"{values['map_at_k']:.6f} | {values['hit_rate_at_k']:.6f} | "
            f"{values['catalog_coverage_at_k']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Metrics are macro-averaged across users. The JSON artifact contains "
            "95% user-bootstrap confidence intervals for NDCG, MAP, and Hit Rate. "
            "This benchmark covers warm users and training-catalog products; cold-start "
            "performance is evaluated separately under issue #11.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "make evaluate-final",
            "```",
            "",
        ]
    )
    tracking = report.get("tracking")
    if tracking:
        lines.extend(
            [
                "## Databricks MLflow",
                "",
                f"- Experiment: `{tracking['experiment_name']}`",
                f"- Experiment ID: `{tracking['experiment_id']}`",
                f"- Run ID: `{tracking['run_id']}`",
                "",
            ]
        )
    return "\n".join(lines)


def evaluate_final(
    processed_dir: Path,
    artifacts_dir: Path,
    report_path: Path,
    max_train_rows: int = 500_000,
    max_eval_users: int = 1_000,
    n_components: int = 64,
    max_text_features: int = 20_000,
    k: int = 12,
    collaborative_weight: float = 0.75,
    include_fresh_catalog_items: bool = False,
    min_fresh_in_top_k: int = 0,
    bootstrap_samples: int = 1_000,
    random_state: int = 42,
    tracking_uri: str | None = None,
    experiment_name: str = "/Shared/reco-nova-final-evaluation",
    run_name: str | None = None,
) -> dict[str, object]:
    """Retrain with frozen choices on train+validation and evaluate test once."""
    paths = {
        "train": processed_dir / "interactions_train.parquet",
        "validation": processed_dir / "interactions_val.parquet",
        "test": processed_dir / "interactions_test.parquet",
        "items": processed_dir / "items_clean.parquet",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run `make preprocess` first.")

    columns = ["customer_id", "article_id"]
    train = pd.read_parquet(paths["train"], columns=columns).dropna()
    validation = pd.read_parquet(paths["validation"], columns=columns).dropna()
    development = pd.concat([train, validation], ignore_index=True)
    development = _positive_limit(development, max_train_rows).astype(str)
    items = pd.read_parquet(paths["items"], columns=["article_id", "item_text"])
    catalog_item_ids = set(items["article_id"].astype(str))
    training_item_ids = set(development["article_id"])
    fresh_item_ids = catalog_item_ids - training_item_ids

    popularity = PopularityRecommender().fit(development)
    collaborative = CollaborativeSVD(n_components, random_state).fit(development)
    content_candidates = None if include_fresh_catalog_items else training_item_ids
    content = ContentRecommender(
        n_components=n_components,
        max_features=max_text_features,
        random_state=random_state,
    ).fit(development, items, candidate_item_ids=content_candidates)
    hybrid = HybridRecommender(
        collaborative,
        content,
        collaborative_weight=collaborative_weight,
        fresh_item_ids=fresh_item_ids,
        min_fresh_in_top_k=min_fresh_in_top_k,
    )

    # Test data is intentionally not used until all models and weights are frozen.
    test = pd.read_parquet(paths["test"], columns=columns).dropna()
    truth = build_ground_truth(
        test,
        known_users=set(collaborative.user_ids_),
        known_items=set(collaborative.item_ids_),
        seen_by_user=popularity.seen_,
    )
    users = np.array(sorted(truth))
    if max_eval_users > 0 and len(users) > max_eval_users:
        rng = np.random.default_rng(random_state)
        users = np.sort(rng.choice(users, size=max_eval_users, replace=False))
    evaluation_users = users.tolist()
    truth = {user: truth[user] for user in evaluation_users}
    catalog_size = len(collaborative.item_ids_)

    metrics = {}
    for name, model in (
        ("popularity", popularity),
        ("collaborative_svd", collaborative),
        ("content_tfidf", content),
        ("hybrid_frozen", hybrid),
    ):
        recommendations = _recommend(model, evaluation_users, k)
        metrics[name] = _evaluate_model(
            recommendations,
            truth,
            catalog_size,
            k,
            bootstrap_samples,
            random_state,
            fresh_item_ids if include_fresh_catalog_items else None,
        )

    report: dict[str, object] = {
        "configuration": {
            "max_train_rows": max_train_rows,
            "max_eval_users": max_eval_users,
            "n_components": n_components,
            "max_text_features": max_text_features,
            "k": k,
            "collaborative_weight": collaborative_weight,
            "include_fresh_catalog_items": include_fresh_catalog_items,
            "min_fresh_in_top_k": min_fresh_in_top_k,
            "bootstrap_samples": bootstrap_samples,
            "random_state": random_state,
        },
        "data": {
            "training_rows": len(development),
            "training_users": development["customer_id"].nunique(),
            "training_items": development["article_id"].nunique(),
            "fresh_catalog_items": len(fresh_item_ids),
            "warm_test_users": len(truth),
        },
        "metrics": metrics,
        "evaluation_scope": (
            "Frozen configuration evaluated once on a seeded warm-user sample "
            "from the untouched test split."
        ),
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(popularity, artifacts_dir / "popularity.joblib")
    joblib.dump(collaborative, artifacts_dir / "collaborative_svd.joblib")
    joblib.dump(content, artifacts_dir / "content_tfidf.joblib")
    json_path = artifacts_dir / "final_evaluation.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    previous_markdown = report_path.read_text() if report_path.exists() else None
    report_path.write_text(_markdown(report), encoding="utf-8")
    if tracking_uri:
        try:
            report["tracking"] = log_recommender_run(
                report,
                artifacts_dir,
                tracking_uri,
                experiment_name,
                run_name,
                task="final-offline-evaluation",
                artifact_path="final-evaluation",
            )
        except Exception:
            if previous_markdown is not None:
                report_path.write_text(previous_markdown, encoding="utf-8")
            raise
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_path.write_text(_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final held-out evaluation")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/final"))
    parser.add_argument(
        "--report-path", type=Path, default=Path("docs/offline_evaluation_report.md")
    )
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--max-eval-users", type=int, default=1_000)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--max-text-features", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--collaborative-weight", type=float, default=0.75)
    parser.add_argument(
        "--include-fresh-catalog-items",
        action="store_true",
        help="Fit content model on full catalog to expose fresh metadata-only items.",
    )
    parser.add_argument("--min-fresh-in-top-k", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--experiment-name",
        default=os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "/Shared/reco-nova-final-evaluation"
        ),
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate_final(**vars(parse_args())), indent=2))


if __name__ == "__main__":
    main()
