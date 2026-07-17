"""Train, tune, and evaluate content and hybrid recommenders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reco_nova.evaluation import (
    OfflineMetrics,
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
from reco_nova.train import _positive_limit, build_ground_truth, read_interactions


def _recommend_users(model: object, users: list[str], k: int) -> dict[str, list[str]]:
    return {
        user: [item for item, _ in model.recommend(user, k)]  # type: ignore[attr-defined]
        for user in users
    }


def _metric_key(metrics: OfflineMetrics) -> tuple[float, float, float]:
    return (metrics.ndcg_at_k, metrics.map_at_k, metrics.hit_rate_at_k)


def _attach_fresh_metrics(
    metric_dict: dict[str, int | float],
    recommendations: dict[str, list[str]],
    fresh_item_ids: set[str],
    k: int,
) -> None:
    """Add fresh-catalog exposure metrics to one model result."""
    metric_dict["fresh_catalog_coverage_at_k"] = fresh_catalog_coverage_at_k(
        recommendations, fresh_item_ids, k
    )
    metric_dict["fresh_share_at_k"] = fresh_share_at_k(
        recommendations, fresh_item_ids, k
    )
    metric_dict["users_with_fresh_hit_at_k"] = users_with_fresh_hit_at_k(
        recommendations, fresh_item_ids, k
    )


def train_hybrid_and_evaluate(
    processed_dir: Path,
    artifacts_dir: Path,
    max_train_rows: int = 0,
    max_eval_users: int = 1_000,
    n_components: int = 64,
    max_text_features: int = 20_000,
    k: int = 12,
    random_state: int = 42,
    hybrid_weights: tuple[float, ...] = (0.25, 0.5, 0.75),
    include_fresh_catalog_items: bool = False,
    min_fresh_in_top_k: int = 0,
    recency_half_life_days: float | None = None,
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
    if min_fresh_in_top_k < 0:
        raise ValueError("min_fresh_in_top_k must be non-negative")

    want_recency = recency_half_life_days is not None
    train, recency_available = read_interactions(paths["train"], want_recency)
    validation, _ = read_interactions(paths["validation"], False)
    items = pd.read_parquet(paths["items"], columns=["article_id", "item_text"])
    train = _positive_limit(train, max_train_rows)
    train["customer_id"] = train["customer_id"].astype(str)
    train["article_id"] = train["article_id"].astype(str)
    effective_recency = recency_half_life_days if recency_available else None
    catalog_item_ids = set(items["article_id"].astype(str))
    training_item_ids = set(train["article_id"])
    fresh_item_ids = catalog_item_ids - training_item_ids

    popularity = PopularityRecommender().fit(train)
    collaborative = CollaborativeSVD(n_components, random_state).fit(
        train, recency_half_life_days=effective_recency
    )
    content_candidates = None if include_fresh_catalog_items else training_item_ids
    content = ContentRecommender(
        n_components=n_components,
        max_features=max_text_features,
        random_state=random_state,
    ).fit(
        train,
        items,
        candidate_item_ids=content_candidates,
        recency_half_life_days=effective_recency,
    )

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
        recommendations = _recommend_users(model, evaluation_users, k)
        result = evaluate_rankings(recommendations, truth, k)
        metrics[name] = result.to_dict()
        if include_fresh_catalog_items:
            _attach_fresh_metrics(metrics[name], recommendations, fresh_item_ids, k)

    best_weight = hybrid_weights[0]
    best_metrics: OfflineMetrics | None = None
    best_recommendations: dict[str, list[str]] | None = None
    tuning: dict[str, dict[str, int | float]] = {}
    for weight in hybrid_weights:
        model = HybridRecommender(
            collaborative,
            content,
            weight,
            fresh_item_ids=fresh_item_ids,
            min_fresh_in_top_k=min_fresh_in_top_k,
        )
        recommendations = _recommend_users(model, evaluation_users, k)
        result = evaluate_rankings(recommendations, truth, k)
        tuning_entry = result.to_dict()
        if include_fresh_catalog_items:
            _attach_fresh_metrics(
                tuning_entry, recommendations, fresh_item_ids, k
            )
        tuning[f"cf_{weight:.2f}"] = tuning_entry
        if best_metrics is None or _metric_key(result) > _metric_key(best_metrics):
            best_weight, best_metrics = weight, result
            best_recommendations = recommendations
    assert best_metrics is not None
    metrics["hybrid_best"] = best_metrics.to_dict()
    if include_fresh_catalog_items and best_recommendations is not None:
        _attach_fresh_metrics(
            metrics["hybrid_best"], best_recommendations, fresh_item_ids, k
        )

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
            "include_fresh_catalog_items": include_fresh_catalog_items,
            "min_fresh_in_top_k": min_fresh_in_top_k,
            "recency_half_life_days": effective_recency,
        },
        "data": {
            "training_rows": len(train),
            "training_users": train["customer_id"].nunique(),
            "training_items": train["article_id"].nunique(),
            "catalog_items": len(content.item_ids_),
            "fresh_catalog_items": len(fresh_item_ids),
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
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-users", type=int, default=1_000)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--max-text-features", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hybrid-weights", type=_weights, default=(0.25, 0.5, 0.75))
    parser.add_argument(
        "--include-fresh-catalog-items",
        action="store_true",
        help="Fit content model on full catalog to expose fresh metadata-only items.",
    )
    parser.add_argument("--min-fresh-in-top-k", type=int, default=0)
    parser.add_argument(
        "--recency-half-life-days",
        type=float,
        default=None,
        help=(
            "Exponentially decay interaction weight with this half-life in "
            "days (requires the event_ts column). Disabled by default."
        ),
    )
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
