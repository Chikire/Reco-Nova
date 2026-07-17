import json

import numpy as np
import pandas as pd

from reco_nova.evaluate_policy_impact import (
    RandomPolicy,
    _lift,
    _simulate_clicks,
    evaluate_policy_impact,
)


def test_random_policy_excludes_seen_items():
    rng = np.random.default_rng(42)
    policy = RandomPolicy(
        item_ids=["a", "b", "c"],
        seen_by_user={"u1": {"a", "b"}},
        rng=rng,
    )

    recs = policy.recommend("u1", k=3)
    recommended_ids = [item for item, _ in recs]

    assert recommended_ids == ["c"]


def test_simulate_clicks_returns_expected_aggregate_fields():
    rng = np.random.default_rng(123)
    recommendations = {
        "u1": ["a", "b"],
        "u2": ["c", "d"],
    }
    truth = {
        "u1": {"a"},
        "u2": {"x"},
    }

    outcomes = _simulate_clicks(
        recommendations,
        truth,
        rounds=10,
        rng=rng,
        relevant_click_base=1.0,
        irrelevant_click_base=0.0,
    ).to_dict()

    assert outcomes["sessions"] == 20
    assert outcomes["impressions"] == 40
    assert outcomes["clicks"] == 10
    assert outcomes["relevant_clicks"] == 10
    assert outcomes["click_through_rate"] == 0.25
    assert outcomes["relevant_click_through_rate"] == 0.25


def test_lift_handles_zero_baseline_and_regular_case():
    assert _lift(0.2, 0.1) == 1.0
    assert _lift(0.0, 0.0) == 0.0
    assert _lift(0.1, 0.0) == float("inf")


def test_evaluate_policy_impact_writes_reports(tmp_path):
    processed = tmp_path / "processed"
    artifacts = tmp_path / "policy_impact"
    report_path = tmp_path / "policy_impact.md"
    processed.mkdir()

    train = pd.DataFrame(
        {
            "customer_id": [
                "u1",
                "u1",
                "u2",
                "u2",
                "u3",
                "u3",
                "u4",
                "u4",
            ],
            "article_id": ["a", "b", "b", "c", "c", "d", "a", "d"],
        }
    )
    validation = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4"],
            "article_id": ["c", "d", "a", "b"],
        }
    )
    test = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4", "new"],
            "article_id": ["d", "a", "b", "c", "a"],
        }
    )
    items = pd.DataFrame(
        {
            "article_id": ["a", "b", "c", "d", "fresh"],
            "item_text": [
                "red cotton shirt",
                "blue cotton shirt",
                "blue denim jeans",
                "black running shoes",
                "green cotton shirt",
            ],
        }
    )

    for name, frame in (
        ("interactions_train.parquet", train),
        ("interactions_val.parquet", validation),
        ("interactions_test.parquet", test),
        ("items_clean.parquet", items),
    ):
        frame.to_parquet(processed / name, index=False)

    report = evaluate_policy_impact(
        processed_dir=processed,
        artifacts_dir=artifacts,
        report_path=report_path,
        baseline_policy="popularity",
        n_components=2,
        max_text_features=100,
        max_eval_users=10,
        k=2,
        hybrid_weights=(0.25, 0.75),
        simulation_rounds=20,
        relevant_click_base=0.5,
        irrelevant_click_base=0.0,
        random_state=42,
    )

    assert report["configuration"]["baseline_policy"] == "popularity"
    assert report["configuration"]["selected_hybrid_weight"] in {0.25, 0.75}
    assert set(report["offline_metrics"]) == {
        "baseline",
        "personalized_hybrid",
    }
    assert set(report["simulated_outcomes"]) == {
        "baseline",
        "personalized_hybrid",
    }
    assert "click_through_rate_lift" in report["lift"]
    assert (artifacts / "policy_impact_report.json").exists()
    assert report_path.exists()

    saved = json.loads((artifacts / "policy_impact_report.json").read_text())
    assert saved["configuration"]["baseline_policy"] == "popularity"
    markdown = report_path.read_text()
    assert "Policy Impact Simulation" in markdown
    assert "Lift (Personalized vs Baseline)" in markdown
