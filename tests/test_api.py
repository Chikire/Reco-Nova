from fastapi.testclient import TestClient
from types import SimpleNamespace

from reco_nova.api import RecommendationService, create_app
from reco_nova.models import ColdStartResult, HybridScore


class StubHybrid:
    collaborative = SimpleNamespace(user_to_index_={})
    content = SimpleNamespace(item_to_index_={})

    def recommend(self, user_id, k=10):
        return [("a", 0.9), ("b", 0.7)][:k]

    def recommend_with_components(self, user_id, k=10):
        return [
            HybridScore("a", 0.9, 0.7, 0.2),
            HybridScore("b", 0.7, 0.5, 0.2),
        ][:k]


class StubColdStart:
    def recommend(self, **kwargs):
        return ColdStartResult(
            "global_popularity", [("b", 2.0)], "Popular products across all shoppers."
        )


def service():
    return RecommendationService(
        hybrid=StubHybrid(),
        cold_start=StubColdStart(),
        known_users={"known"},
        metadata={
            "a": {"prod_name": "Shirt", "product_group_name": "Tops", "image_path": None},
            "b": {"prod_name": "Jeans", "product_group_name": "Trousers", "image_path": "b.jpg"},
        },
    )


def test_health_and_known_user_recommendation():
    with TestClient(create_app(service())) as client:
        root = client.get("/")
        health = client.get("/health")
        response = client.post("/recommend", json={"user_id": "known", "limit": 1})
    assert root.json()["documentation"] == "/docs"
    assert health.json()["models_ready"] is True
    assert response.status_code == 200
    assert response.json()["strategy"] == "hybrid_personalized"
    assert response.json()["recommendations"][0]["product_name"] == "Shirt"
    assert response.json()["recommendations"][0]["signals"] == {
        "collaborative": 0.7,
        "content": 0.2,
    }


def test_unknown_user_routes_to_cold_start_and_explain():
    with TestClient(create_app(service())) as client:
        response = client.post("/explain", json={"user_id": "new", "age": 20})
    assert response.status_code == 200
    assert response.json()["strategy"] == "global_popularity"
    assert response.json()["recommendations"][0]["reason"]


def test_request_validation_rejects_invalid_limit():
    with TestClient(create_app(service())) as client:
        response = client.post("/recommend", json={"limit": 0})
    assert response.status_code == 422


def test_assistant_endpoint_returns_catalog_grounded_products(monkeypatch):
    monkeypatch.setenv("RECO_NOVA_LLM_PROVIDER", "local")
    with TestClient(create_app(service())) as client:
        response = client.post(
            "/assistant/chat",
            json={"message": "Show me one Tops product", "user_id": "known"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "local"
    assert body["intent"]["product_group"] == "Tops"
    assert body["recommendations"][0]["article_id"] == "a"


def test_assistant_budget_query_still_returns_products(monkeypatch):
    """Budget-qualified queries must not return 0 results.

    Regression: 'Show me tops under $50' was returning 0 grounded
    recommendations because the budget tokens ('under', '50') were not
    in the stopword list, causing them to be used as taxonomy tokens that
    matched no product type.  The catalog search must now either match via
    the remaining tokens or fall back to the recommendation service.
    """
    monkeypatch.setenv("RECO_NOVA_LLM_PROVIDER", "local")
    with TestClient(create_app(service())) as client:
        response = client.post(
            "/assistant/chat",
            json={"message": "Show me tops under $50"},
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) > 0, (
        "Budget query must not return 0 results; catalog search or fallback should populate"
    )
    # Budget is now applied server-side via prices; it should not appear as unsupported.
    assert "budget" not in body["unsupported_constraints"]


def test_missing_artifacts_report_degraded_and_503(monkeypatch, tmp_path):
    monkeypatch.setenv("RECO_NOVA_ARTIFACTS_DIR", str(tmp_path / "missing"))
    with TestClient(create_app()) as client:
        health = client.get("/health")
        response = client.post("/recommend", json={})
    assert health.json()["status"] == "degraded"
    assert response.status_code == 503
