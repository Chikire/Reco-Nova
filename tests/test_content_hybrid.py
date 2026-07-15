import pandas as pd
import pytest

from reco_nova.models import ContentRecommender, HybridRecommender
from reco_nova.models.hybrid import _minmax


class StubRecommender:
    def __init__(self, results):
        self.results = results

    def recommend(self, user_id, k=10):
        return self.results[:k]


def test_content_recommender_excludes_history_and_handles_unknown_user():
    items = pd.DataFrame(
        {
            "article_id": ["red1", "red2", "blue1", "shoe1"],
            "item_text": [
                "red cotton summer shirt",
                "red linen casual shirt",
                "blue denim winter jacket",
                "black leather running shoe",
            ],
        }
    )
    interactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2", "u2"],
            "article_id": ["red1", "blue1", "shoe1", "blue1"],
        }
    )
    model = ContentRecommender(n_components=2).fit(interactions, items)
    output = [item for item, _ in model.recommend("u1", 2)]
    assert not ({"red1", "blue1"} & set(output))
    assert len(model.recommend("new", 2)) == 2


def test_hybrid_normalizes_and_blends_candidate_scores():
    collaborative = StubRecommender([("a", 100.0), ("b", 50.0)])
    content = StubRecommender([("b", 0.9), ("c", 0.8)])
    model = HybridRecommender(collaborative, content, collaborative_weight=0.5)
    output = model.recommend("u1", 3)
    assert output[0][0] == "a"
    assert {item for item, _ in output} == {"a", "b", "c"}
    assert _minmax({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}


def test_hybrid_rejects_invalid_weight():
    with pytest.raises(ValueError):
        HybridRecommender(StubRecommender([]), StubRecommender([]), 1.1)
