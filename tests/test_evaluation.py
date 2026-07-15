import pytest

from reco_nova.evaluation import (
    average_precision_at_k,
    evaluate_rankings,
    hit_rate_at_k,
    ndcg_at_k,
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
