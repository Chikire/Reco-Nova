from fastapi.testclient import TestClient

from reco_nova.api import RecommendationService, create_app
from reco_nova.models import ColdStartResult


class StubHybrid:
    def recommend(self, user_id, k=10):
        return [("a", 0.9), ("b", 0.7)][:k]


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


def test_missing_artifacts_report_degraded_and_503(monkeypatch, tmp_path):
    monkeypatch.setenv("RECO_NOVA_ARTIFACTS_DIR", str(tmp_path / "missing"))
    with TestClient(create_app()) as client:
        health = client.get("/health")
        response = client.post("/recommend", json={})
    assert health.json()["status"] == "degraded"
    assert response.status_code == 503
