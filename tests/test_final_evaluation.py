import json

import pandas as pd

from reco_nova.evaluate_final import evaluate_final


def test_final_evaluation_writes_json_markdown_and_models(tmp_path):
    processed = tmp_path / "processed"
    artifacts = tmp_path / "final"
    report_path = tmp_path / "report.md"
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
    test = pd.DataFrame(
        {"customer_id": ["u1", "u2", "u3", "new"], "article_id": ["d", "a", "b", "a"]}
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
    for name, frame in (
        ("interactions_train.parquet", train),
        ("interactions_val.parquet", validation),
        ("interactions_test.parquet", test),
        ("items_clean.parquet", items),
    ):
        frame.to_parquet(processed / name, index=False)

    report = evaluate_final(
        processed,
        artifacts,
        report_path,
        n_components=2,
        max_text_features=100,
        max_eval_users=10,
        k=2,
        bootstrap_samples=50,
    )

    assert report["data"]["warm_test_users"] == 3
    assert set(report["metrics"]) == {
        "popularity",
        "collaborative_svd",
        "content_tfidf",
        "hybrid_frozen",
    }
    assert "Catalog Coverage@K" in report_path.read_text()
    saved = json.loads((artifacts / "final_evaluation.json").read_text())
    assert saved["configuration"]["collaborative_weight"] == 0.75
    assert (artifacts / "content_tfidf.joblib").exists()
