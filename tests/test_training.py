import json

import pandas as pd

from reco_nova.train import build_ground_truth, train_and_evaluate


def test_ground_truth_excludes_cold_and_previously_seen_items():
    holdout = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "new"],
            "article_id": ["a", "b", "b"],
        }
    )
    truth = build_ground_truth(
        holdout,
        known_users={"u1"},
        known_items={"a", "b"},
        seen_by_user={"u1": {"a"}},
    )
    assert truth == {"u1": {"b"}}


def test_training_pipeline_writes_models_and_report(tmp_path):
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    processed.mkdir()
    train = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "article_id": ["a", "b", "b", "c", "c", "d"],
        }
    )
    validation = pd.DataFrame(
        {"customer_id": ["u1", "u2", "new"], "article_id": ["c", "d", "a"]}
    )
    train.to_parquet(processed / "interactions_train.parquet", index=False)
    validation.to_parquet(processed / "interactions_val.parquet", index=False)

    report = train_and_evaluate(
        processed, artifacts, n_components=2, max_eval_users=10, k=2
    )

    assert report["data"]["warm_validation_users"] == 2
    assert (artifacts / "popularity.joblib").exists()
    assert (artifacts / "collaborative_svd.joblib").exists()
    saved = json.loads((artifacts / "baseline_metrics.json").read_text())
    assert set(saved["metrics"]) == {"popularity", "collaborative_svd"}
