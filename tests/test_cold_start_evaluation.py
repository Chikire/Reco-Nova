import pandas as pd

from reco_nova.evaluate_cold_start import evaluate_cold_start


def test_cold_start_evaluation_identifies_unseen_users_and_writes_report(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    report_path = tmp_path / "cold.md"
    processed.mkdir()
    train = pd.DataFrame({"customer_id": ["u1", "u1", "u2", "u2"], "article_id": ["a", "b", "b", "c"]})
    val = pd.DataFrame({"customer_id": ["u1"], "article_id": ["c"]})
    test = pd.DataFrame({"customer_id": ["new1", "new2", "u1"], "article_id": ["a", "b", "b"]})
    customers = pd.DataFrame({"customer_id": ["u1", "u2", "new1", "new2"], "age": [20, 20, 20, 55], "club_member_status": ["active", "active", "active", "none"]})
    items = pd.DataFrame({"article_id": ["a", "b", "c"], "item_text": ["red cotton shirt", "blue cotton shirt", "black denim jeans"], "product_group_name": ["tops", "tops", "trousers"]})
    for name, frame in [("interactions_train.parquet", train), ("interactions_val.parquet", val), ("interactions_test.parquet", test), ("customers_clean.parquet", customers), ("items_clean.parquet", items)]:
        frame.to_parquet(processed / name, index=False)
    report = evaluate_cold_start(processed, artifacts, report_path, n_components=2, max_text_features=20, min_segment_events=1, max_eval_users=10, k=2, bootstrap_samples=20)
    assert report["data"]["new_test_users_available"] == 2
    assert set(report["metrics"]) == {"global_popularity", "demographic_fallback"}
    assert {example["strategy"] for example in report["examples"]} == {"session_content", "demographic_popularity", "category_popularity", "global_popularity"}
    assert report_path.exists()
