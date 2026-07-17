"""Train and evaluate Reco-Nova baseline recommenders.

Run with ``python -m reco_nova.train --help``. Training uses all eligible rows
by default; pass a positive ``--max-train-rows`` value to cap training data
for faster local iteration. ``--max-eval-users`` still defaults to a capped
sample of warm validation users to keep evaluation fast.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq

from reco_nova.evaluation import evaluate_rankings
from reco_nova.models import CollaborativeSVD, PopularityRecommender
from reco_nova.tracking import log_baseline_run


def _positive_limit(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    return frame.tail(limit).copy() if limit > 0 and len(frame) > limit else frame


def _parquet_columns(path: Path) -> set[str]:
    """Return the column names of a parquet file without reading its data."""
    try:
        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return set()


def read_interactions(
    path: Path, want_recency: bool = False
) -> tuple[pd.DataFrame, bool]:
    """Read customer/article interactions, adding ``event_ts`` when recency
    weighting is requested and the column is available in the parquet file.

    Returns the frame plus whether recency weighting can actually be applied.
    """
    columns = ["customer_id", "article_id"]
    has_recency = False
    if want_recency and "event_ts" in _parquet_columns(path):
        columns.append("event_ts")
        has_recency = True
    frame = pd.read_parquet(path, columns=columns).dropna()
    return frame, has_recency


def build_ground_truth(
    holdout: pd.DataFrame,
    known_users: set[str],
    known_items: set[str],
    seen_by_user: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    """Build warm-start relevance sets supported by collaborative filtering.

    Holdout items seen by the same user during training are removed because the
    recommenders intentionally do not return already-purchased products.
    """
    frame = holdout[["customer_id", "article_id"]].dropna().astype(str)
    frame = frame[
        frame["customer_id"].isin(known_users)
        & frame["article_id"].isin(known_items)
    ]
    truth = frame.groupby("customer_id")["article_id"].agg(set).to_dict()
    if seen_by_user:
        truth = {
            user: items - seen_by_user.get(user, set())
            for user, items in truth.items()
        }
    return {user: items for user, items in truth.items() if items}


def train_and_evaluate(
    processed_dir: Path,
    artifacts_dir: Path,
    max_train_rows: int = 0,
    max_eval_users: int = 1_000,
    n_components: int = 64,
    k: int = 12,
    random_state: int = 42,
    recency_half_life_days: float | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "/Shared/reco-nova-baselines",
    run_name: str | None = None,
) -> dict[str, object]:
    """Train popularity/SVD models and persist models plus a metric report."""
    train_path = processed_dir / "interactions_train.parquet"
    validation_path = processed_dir / "interactions_val.parquet"
    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `make preprocess` before training."
            )

    want_recency = recency_half_life_days is not None
    train, recency_available = read_interactions(train_path, want_recency)
    validation, _ = read_interactions(validation_path, False)
    train = _positive_limit(train, max_train_rows)
    train["customer_id"] = train["customer_id"].astype(str)
    train["article_id"] = train["article_id"].astype(str)
    effective_recency = recency_half_life_days if recency_available else None

    popularity = PopularityRecommender().fit(train)
    collaborative = CollaborativeSVD(
        n_components=n_components, random_state=random_state
    ).fit(train, recency_half_life_days=effective_recency)

    truth = build_ground_truth(
        validation,
        known_users=set(collaborative.user_ids_),
        known_items=set(collaborative.item_ids_),
        seen_by_user=popularity.seen_,
    )
    users = sorted(truth)
    if max_eval_users > 0:
        users = users[:max_eval_users]
    truth = {user: truth[user] for user in users}

    # Request extra candidates because training items are excluded by both models.
    predictions = {}
    for name, model in (("popularity", popularity), ("collaborative_svd", collaborative)):
        predictions[name] = {
            user: [item for item, _ in model.recommend(user, k)] for user in users
        }

    metrics = {
        name: evaluate_rankings(output, truth, k).to_dict()
        for name, output in predictions.items()
    }
    report: dict[str, object] = {
        "configuration": {
            "max_train_rows": max_train_rows,
            "max_eval_users": max_eval_users,
            "n_components": n_components,
            "k": k,
            "random_state": random_state,
            "recency_half_life_days": effective_recency,
        },
        "data": {
            "training_rows": len(train),
            "training_users": train["customer_id"].nunique(),
            "training_items": train["article_id"].nunique(),
            "warm_validation_users": len(truth),
        },
        "metrics": metrics,
        "evaluation_scope": (
            "Warm-start validation users and items only; duplicate holdout "
            "purchases are one relevant item and items previously purchased "
            "by that user are excluded."
        ),
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(popularity, artifacts_dir / "popularity.joblib")
    joblib.dump(collaborative, artifacts_dir / "collaborative_svd.joblib")
    (artifacts_dir / "baseline_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if tracking_uri:
        tracking = log_baseline_run(
            report=report,
            artifacts_dir=artifacts_dir,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
        )
        report["tracking"] = tracking
        # Keep the local report connected to its remote Databricks run.
        (artifacts_dir / "baseline_metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline recommenders")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-users", type=int, default=1_000)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--recency-half-life-days",
        type=float,
        default=None,
        help=(
            "Exponentially decay interaction weight with this half-life in "
            "days (requires the event_ts column). Disabled by default."
        ),
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow server URI; use 'databricks' for a configured workspace",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/reco-nova-baselines"),
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_and_evaluate(**vars(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
