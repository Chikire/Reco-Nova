import pandas as pd

from reco_nova.models import CollaborativeSVD, PopularityRecommender


def interactions():
    return pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u3"],
            "article_id": ["a", "b", "b", "c", "c", "d", "d"],
        }
    )


def test_popularity_excludes_seen_items_and_supports_unknown_users():
    model = PopularityRecommender().fit(interactions())
    known = [item for item, _ in model.recommend("u3", 3)]
    unknown = [item for item, _ in model.recommend("new", 2)]
    assert not ({"c", "d"} & set(known))
    assert unknown == ["b", "c"]


def test_collaborative_svd_returns_unseen_items_and_popularity_fallback():
    model = CollaborativeSVD(n_components=2).fit(interactions())
    known = [item for item, _ in model.recommend("u1", 2)]
    fallback = model.recommend("new", 2)
    assert len(known) == 2
    assert not ({"a", "b"} & set(known))
    assert fallback == model.popularity_.recommend("new", 2)
