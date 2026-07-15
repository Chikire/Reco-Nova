from reco_nova.tracking import _numeric_metrics


def test_numeric_metrics_flattens_models_and_omits_counts():
    report = {
        "metrics": {
            "popularity": {
                "k": 12,
                "users_evaluated": 100,
                "ndcg_at_k": 0.25,
            },
            "collaborative_svd": {
                "k": 12,
                "users_evaluated": 100,
                "hit_rate_at_k": 0.4,
            },
        }
    }
    assert _numeric_metrics(report) == {
        "popularity.ndcg_at_k": 0.25,
        "collaborative_svd.hit_rate_at_k": 0.4,
    }
