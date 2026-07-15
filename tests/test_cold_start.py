import pandas as pd

from reco_nova.models import ColdStartRecommender, ContentRecommender
from reco_nova.models.cold_start import age_band


def fixtures():
    interactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "article_id": ["a", "b", "a", "c", "d", "c"],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3"],
            "age": [20, 22, 55],
            "club_member_status": ["active", "active", "none"],
        }
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
            "product_group_name": ["tops", "tops", "trousers", "shoes"],
        }
    )
    return interactions, customers, items


def test_age_bands_cover_missing_and_boundaries():
    assert age_band(None) == "unknown"
    assert age_band(24) == "16-24"
    assert age_band(25) == "25-34"
    assert age_band(50) == "50+"


def test_cold_start_fallback_hierarchy():
    interactions, customers, items = fixtures()
    content = ContentRecommender(n_components=2).fit(interactions, items)
    model = ColdStartRecommender(min_segment_events=2).fit(
        interactions, customers, items, content
    )

    session = model.recommend(k=2, session_article_ids=["a"], age=20)
    demographic = model.recommend(k=2, age=20, club_member_status="active")
    category = model.recommend(
        k=2, preferred_product_group="shoes", use_demographics=False
    )
    global_result = model.recommend(k=2, use_demographics=False)

    assert session.strategy == "session_content"
    assert "a" not in {item for item, _ in session.recommendations}
    assert demographic.strategy == "demographic_popularity"
    assert category.strategy == "category_popularity"
    assert category.recommendations[0][0] == "d"
    assert global_result.strategy == "global_popularity"
