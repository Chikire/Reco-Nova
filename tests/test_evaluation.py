import pytest

from reco_nova.evaluation import (
    average_precision_at_k,
    bootstrap_metric_intervals,
    catalog_coverage,
    evaluate_rankings,
    fresh_catalog_coverage_at_k,
    fresh_share_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    users_with_fresh_hit_at_k,
)


def test_perfect_ranking_scores_one():
    recommended = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert ndcg_at_k(recommended, relevant, 2) == pytest.approx(1.0)
    assert average_precision_at_k(recommended, relevant, 2) == pytest.approx(1.0)
    assert hit_rate_at_k(recommended, relevant, 2) == 1.0


def test_metrics_ignore_duplicates_and_respect_k():
    assert hit_rate_at_k(["x", "x", "a"], {"a"}, 2) == 1.0
    assert average_precision_at_k(["x", "a", "b"], {"a", "b"}, 2) == 0.25


def test_evaluate_rankings_uses_common_users_with_truth():
    metrics = evaluate_rankings(
        {"u1": ["a"], "u2": ["z"], "extra": ["a"]},
        {"u1": {"a"}, "u2": {"b"}, "empty": set()},
        k=1,
    )
    assert metrics.users_evaluated == 2
    assert metrics.hit_rate_at_k == 0.5


def test_catalog_coverage_counts_unique_top_k_items():
    recommendations = {"u1": ["a", "b"], "u2": ["b", "c"]}
    assert catalog_coverage(recommendations, catalog_size=6, k=2) == 0.5


def test_fresh_item_exposure_metrics():
    recommendations = {
        "u1": ["a", "fresh1"],
        "u2": ["fresh1", "fresh2"],
        "u3": ["a", "b"],
    }
    fresh_ids = {"fresh1", "fresh2", "fresh3"}
    assert fresh_catalog_coverage_at_k(recommendations, fresh_ids, 2) == pytest.approx(
        2 / 3
    )
    assert fresh_share_at_k(recommendations, fresh_ids, 2) == pytest.approx(
        (0.5 + 1.0 + 0.0) / 3
    )
    assert users_with_fresh_hit_at_k(recommendations, fresh_ids, 2) == pytest.approx(
        2 / 3
    )


def test_bootstrap_intervals_are_reproducible_and_bound_point_estimate():
    recommendations = {"u1": ["a"], "u2": ["x"], "u3": ["c"]}
    truth = {"u1": {"a"}, "u2": {"b"}, "u3": {"c"}}
    first = bootstrap_metric_intervals(
        recommendations, truth, k=1, samples=200, random_state=7
    )
    second = bootstrap_metric_intervals(
        recommendations, truth, k=1, samples=200, random_state=7
    )
    assert first == second
    lower, upper = first["hit_rate_at_k"]
    assert lower <= 2 / 3 <= upper
