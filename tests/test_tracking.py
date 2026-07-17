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


def test_numeric_metrics_includes_hybrid_weight_sweep():
    report = {
        "metrics": {},
        "hybrid_weight_tuning": {
            "cf_0.50": {"k": 12, "users_evaluated": 100, "ndcg_at_k": 0.3}
        },
    }
    assert _numeric_metrics(report) == {"tuning.cf_0.50.ndcg_at_k": 0.3}
