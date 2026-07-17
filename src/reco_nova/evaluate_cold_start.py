"""Proxy evaluation and examples for new-user recommendation fallbacks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from reco_nova.evaluation import (
    bootstrap_metric_intervals,
    catalog_coverage,
    evaluate_rankings,
)
from reco_nova.models import ColdStartRecommender, ContentRecommender
from reco_nova.tracking import log_recommender_run
from reco_nova.train import _positive_limit


def _score(
    recommendations: dict[str, list[str]],
    truth: dict[str, set[str]],
    catalog_size: int,
    k: int,
    bootstrap_samples: int,
    random_state: int,
) -> dict[str, int | float]:
    output = evaluate_rankings(recommendations, truth, k).to_dict()
    output["catalog_coverage_at_k"] = catalog_coverage(
        recommendations, catalog_size, k
    )
    for metric, (lower, upper) in bootstrap_metric_intervals(
        recommendations,
        truth,
        k=k,
        samples=bootstrap_samples,
        random_state=random_state,
    ).items():
        output[f"{metric}_ci95_lower"] = lower
        output[f"{metric}_ci95_upper"] = upper
    return output


def _examples(
    model: ColdStartRecommender,
    k: int,
) -> list[dict[str, object]]:
    band, membership = next(iter(model.segment_rankings_))
    example_age = {"16-24": 20, "25-34": 30, "35-49": 40, "50+": 55}.get(
        band
    )
    category = next(iter(model.category_rankings_))
    session_item = str(model.content_.item_ids_[0])
    cases = [
        (
            "demographics",
            dict(
                age=example_age,
                club_member_status=membership,
            ),
        ),
        (
            "category",
            dict(preferred_product_group=category, use_demographics=False),
        ),
        ("session", dict(session_article_ids=[session_item])),
        ("no_context", dict(use_demographics=False)),
    ]
    output = []
    for case, kwargs in cases:
        result = model.recommend(k=k, **kwargs)
        output.append(
            {
                "case": case,
                "inputs": kwargs,
                "strategy": result.strategy,
                "explanation": result.explanation,
                "article_ids": [item for item, _ in result.recommendations],
            }
        )
    return output


def _markdown(report: dict[str, object]) -> str:
    data, metrics = report["data"], report["metrics"]
    lines = [
        "# Reco-Nova Cold-Start Evaluation",
        "",
        "This proxy benchmark evaluates users whose IDs never appear in train or validation.",
        "Test purchases are used only as relevance labels.",
        "",
        "## Evaluation setup",
        "",
        f"- New test users available: {data['new_test_users_available']:,}",
        f"- New test users evaluated: {data['new_test_users_evaluated']:,}",
        f"- Eligible catalog items: {data['catalog_items']:,}",
        "",
        "## Results",
        "",
        "| Strategy | NDCG@K | MAP@K | Hit Rate@K | Coverage@K |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['ndcg_at_k']:.6f} | {values['map_at_k']:.6f} | "
            f"{values['hit_rate_at_k']:.6f} | {values['catalog_coverage_at_k']:.6f} |"
        )
    lines.extend(["", "## Example outputs", ""])
    for example in report["examples"]:
        lines.extend(
            [
                f"### {example['case']}",
                "",
                f"- Strategy: `{example['strategy']}`",
                f"- Explanation: {example['explanation']}",
                f"- Products: `{', '.join(example['article_ids'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Demographic performance is compared directly with the no-context global "
            "fallback. Session and category strategies are demonstrated separately because "
            "using test purchases to manufacture those inputs would leak relevance labels.",
            "",
            "```bash",
            "make evaluate-cold-start",
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


def evaluate_cold_start(
    processed_dir: Path,
    artifacts_dir: Path,
    report_path: Path,
    max_train_rows: int = 0,
    max_eval_users: int = 1_000,
    n_components: int = 64,
    max_text_features: int = 20_000,
    min_segment_events: int = 50,
    k: int = 12,
    bootstrap_samples: int = 1_000,
    random_state: int = 42,
    tracking_uri: str | None = None,
    experiment_name: str = "/Shared/reco-nova-cold-start",
    run_name: str | None = None,
) -> dict[str, object]:
    paths = {
        name: processed_dir / filename
        for name, filename in {
            "train": "interactions_train.parquet",
            "validation": "interactions_val.parquet",
            "test": "interactions_test.parquet",
            "customers": "customers_clean.parquet",
            "items": "items_clean.parquet",
        }.items()
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run `make preprocess` first.")
    columns = ["customer_id", "article_id"]
    train = pd.read_parquet(paths["train"], columns=columns).dropna().astype(str)
    validation = pd.read_parquet(paths["validation"], columns=columns).dropna().astype(str)
    known_users = set(train["customer_id"]) | set(validation["customer_id"])
    development = _positive_limit(
        pd.concat([train, validation], ignore_index=True), max_train_rows
    )
    customers = pd.read_parquet(
        paths["customers"],
        columns=["customer_id", "age", "club_member_status"],
    )
    items = pd.read_parquet(
        paths["items"],
        columns=["article_id", "item_text", "product_group_name"],
    )
    content = ContentRecommender(
        n_components=n_components,
        max_features=max_text_features,
        random_state=random_state,
    ).fit(development, items, candidate_item_ids=set(development["article_id"]))
    model = ColdStartRecommender(min_segment_events).fit(
        development, customers, items, content
    )

    test = pd.read_parquet(paths["test"], columns=columns).dropna().astype(str)
    catalog = set(model.global_.item_ids_)
    cold = test[~test["customer_id"].isin(known_users)]
    cold = cold[cold["article_id"].isin(catalog)]
    truth = cold.groupby("customer_id")["article_id"].agg(set).to_dict()
    available_users = len(truth)
    users = np.array(sorted(truth))
    if max_eval_users > 0 and len(users) > max_eval_users:
        users = np.sort(
            np.random.default_rng(random_state).choice(
                users, max_eval_users, replace=False
            )
        )
    users = users.tolist()
    truth = {user: truth[user] for user in users}
    customer_lookup = customers.assign(
        customer_id=customers["customer_id"].astype(str)
    ).set_index("customer_id")
    global_recs, demographic_recs = {}, {}
    for user in users:
        global_recs[user] = [
            item
            for item, _ in model.recommend(k=k, use_demographics=False).recommendations
        ]
        row = customer_lookup.loc[user] if user in customer_lookup.index else None
        result = model.recommend(
            k=k,
            age=None if row is None else row["age"],
            club_member_status=None if row is None else row["club_member_status"],
        )
        demographic_recs[user] = [item for item, _ in result.recommendations]

    metrics = {
        "global_popularity": _score(
            global_recs, truth, len(catalog), k, bootstrap_samples, random_state
        ),
        "demographic_fallback": _score(
            demographic_recs, truth, len(catalog), k, bootstrap_samples, random_state
        ),
    }
    report: dict[str, object] = {
        "configuration": {
            "max_train_rows": max_train_rows,
            "max_eval_users": max_eval_users,
            "n_components": n_components,
            "max_text_features": max_text_features,
            "min_segment_events": min_segment_events,
            "k": k,
            "bootstrap_samples": bootstrap_samples,
            "random_state": random_state,
        },
        "data": {
            "training_rows": len(development),
            "catalog_items": len(catalog),
            "new_test_users_available": available_users,
            "new_test_users_evaluated": len(users),
        },
        "metrics": metrics,
        "examples": _examples(model, min(k, 5)),
        "evaluation_scope": "Users absent from full train and validation; known catalog items only.",
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "cold_start.joblib")
    json_path = artifacts_dir / "cold_start_metrics.json"
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
                task="cold-start-evaluation",
                artifact_path="cold-start",
            )
        except Exception:
            if previous_markdown is not None:
                report_path.write_text(previous_markdown, encoding="utf-8")
            raise
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_path.write_text(_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cold-start fallbacks")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/cold_start"))
    parser.add_argument("--report-path", type=Path, default=Path("docs/cold_start_report.md"))
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-users", type=int, default=1_000)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--max-text-features", type=int, default=20_000)
    parser.add_argument("--min-segment-events", type=int, default=50)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/reco-nova-cold-start"),
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate_cold_start(**vars(parse_args())), indent=2))


if __name__ == "__main__":
    main()
