import json

import pandas as pd

from reco_nova.train_hybrid import train_hybrid_and_evaluate


def test_hybrid_training_writes_comparison_and_best_config(tmp_path):
    processed = tmp_path / "processed"
    artifacts = tmp_path / "hybrid"
    processed.mkdir()
    train = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "article_id": ["a", "b", "b", "c", "c", "d"],
        }
    )
    validation = pd.DataFrame(
        {"customer_id": ["u1", "u2", "u3"], "article_id": ["c", "d", "a"]}
    )
    items = pd.DataFrame(
        {
            "article_id": ["a", "b", "c", "d"],
            "item_text": [
                "red cotton shirt",
                "blue cotton shirt",
                "blue denim jeans",
                "black running shoes",
            ],
        }
    )
    train.to_parquet(processed / "interactions_train.parquet", index=False)
    validation.to_parquet(processed / "interactions_val.parquet", index=False)
    items.to_parquet(processed / "items_clean.parquet", index=False)

    report = train_hybrid_and_evaluate(
        processed,
        artifacts,
        n_components=2,
        max_text_features=100,
        max_eval_users=10,
        k=2,
        hybrid_weights=(0.25, 0.75),
    )

    assert set(report["metrics"]) == {
        "popularity",
        "collaborative_svd",
        "content_tfidf",
        "hybrid_best",
    }
    assert report["configuration"]["best_collaborative_weight"] in {0.25, 0.75}
    assert (artifacts / "content_tfidf.joblib").exists()
    config = json.loads((artifacts / "best_hybrid_config.json").read_text())
    assert config["collaborative_weight"] in {0.25, 0.75}
