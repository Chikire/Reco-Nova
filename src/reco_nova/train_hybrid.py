"""Train, tune, and evaluate content and hybrid recommenders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reco_nova.evaluation import OfflineMetrics, evaluate_rankings
from reco_nova.models import (
    CollaborativeSVD,
    ContentRecommender,
    HybridRecommender,
    PopularityRecommender,
)
from reco_nova.tracking import log_recommender_run
from reco_nova.train import _positive_limit, build_ground_truth


def _recommend_users(model: object, users: list[str], k: int) -> dict[str, list[str]]:
    return {
        user: [item for item, _ in model.recommend(user, k)]  # type: ignore[attr-defined]
        for user in users
    }


def _metric_key(metrics: OfflineMetrics) -> tuple[float, float, float]:
    return (metrics.ndcg_at_k, metrics.map_at_k, metrics.hit_rate_at_k)


def train_hybrid_and_evaluate(
    processed_dir: Path,
    artifacts_dir: Path,
    max_train_rows: int = 500_000,
    max_eval_users: int = 1_000,
    n_components: int = 64,
    max_text_features: int = 20_000,
    k: int = 12,
    random_state: int = 42,
    hybrid_weights: tuple[float, ...] = (0.25, 0.5, 0.75),
    tracking_uri: str | None = None,
    experiment_name: str = "/Shared/reco-nova-hybrid",
    run_name: str | None = None,
) -> dict[str, object]:
    """Fit standalone models, tune hybrid weight, and persist the best ensemble."""
    paths = {
        "train": processed_dir / "interactions_train.parquet",
        "validation": processed_dir / "interactions_val.parquet",
        "items": processed_dir / "items_clean.parquet",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run `make preprocess` first.")
    if not hybrid_weights or any(not 0.0 <= weight <= 1.0 for weight in hybrid_weights):
        raise ValueError("hybrid_weights must contain values between zero and one")

    columns = ["customer_id", "article_id"]
    train = pd.read_parquet(paths["train"], columns=columns).dropna()
    validation = pd.read_parquet(paths["validation"], columns=columns).dropna()
    items = pd.read_parquet(paths["items"], columns=["article_id", "item_text"])
    train = _positive_limit(train, max_train_rows).astype(str)

    popularity = PopularityRecommender().fit(train)
    collaborative = CollaborativeSVD(n_components, random_state).fit(train)
    content = ContentRecommender(
        n_components=n_components,
        max_features=max_text_features,
        random_state=random_state,
    ).fit(train, items, candidate_item_ids=set(train["article_id"]))

    truth = build_ground_truth(
        validation,
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

    metrics: dict[str, dict[str, int | float]] = {}
    for name, model in (
        ("popularity", popularity),
        ("collaborative_svd", collaborative),
        ("content_tfidf", content),
    ):
        result = evaluate_rankings(_recommend_users(model, evaluation_users, k), truth, k)
        metrics[name] = result.to_dict()

    best_weight = hybrid_weights[0]
    best_metrics: OfflineMetrics | None = None
    tuning: dict[str, dict[str, int | float]] = {}
    for weight in hybrid_weights:
        model = HybridRecommender(collaborative, content, weight)
        result = evaluate_rankings(_recommend_users(model, evaluation_users, k), truth, k)
        tuning[f"cf_{weight:.2f}"] = result.to_dict()
        if best_metrics is None or _metric_key(result) > _metric_key(best_metrics):
            best_weight, best_metrics = weight, result
    assert best_metrics is not None
    metrics["hybrid_best"] = best_metrics.to_dict()

    report: dict[str, object] = {
        "configuration": {
            "max_train_rows": max_train_rows,
            "max_eval_users": max_eval_users,
            "n_components": n_components,
            "max_text_features": max_text_features,
            "k": k,
            "random_state": random_state,
            "hybrid_weights": ",".join(map(str, hybrid_weights)),
            "best_collaborative_weight": best_weight,
        },
        "data": {
            "training_rows": len(train),
            "training_users": train["customer_id"].nunique(),
            "training_items": train["article_id"].nunique(),
            "catalog_items": len(content.item_ids_),
            "warm_validation_users": len(truth),
        },
        "metrics": metrics,
        "hybrid_weight_tuning": tuning,
        "evaluation_scope": (
            "Seeded random sample of warm-start validation users and items; "
            "previously purchased products are excluded."
        ),
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(popularity, artifacts_dir / "popularity.joblib")
    joblib.dump(collaborative, artifacts_dir / "collaborative_svd.joblib")
    joblib.dump(content, artifacts_dir / "content_tfidf.joblib")
    (artifacts_dir / "best_hybrid_config.json").write_text(
        json.dumps({"collaborative_weight": best_weight}, indent=2), encoding="utf-8"
    )
    report_path = artifacts_dir / "hybrid_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if tracking_uri:
        report["tracking"] = log_recommender_run(
            report,
            artifacts_dir,
            tracking_uri,
            experiment_name,
            run_name,
            task="hybrid-recommender",
            artifact_path="hybrid",
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _weights(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weights must be comma-separated numbers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and tune hybrid recommender")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/hybrid"))
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--max-eval-users", type=int, default=1_000)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--max-text-features", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hybrid-weights", type=_weights, default=(0.25, 0.5, 0.75))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/reco-nova-hybrid"),
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    report = train_hybrid_and_evaluate(**vars(parse_args()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
