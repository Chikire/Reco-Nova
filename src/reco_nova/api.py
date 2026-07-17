"""FastAPI serving layer for personalized and cold-start recommendations."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from reco_nova.models import ColdStartRecommender, HybridRecommender


class RecommendationRequest(BaseModel):
    """Known-user ID plus optional anonymous cold-start context."""

    user_id: str | None = Field(default=None, min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    age: float | None = Field(default=None, ge=13, le=120)
    club_member_status: str | None = None
    preferred_product_group: str | None = None
    session_article_ids: list[str] = Field(default_factory=list, max_length=50)


class ProductRecommendation(BaseModel):
    article_id: str
    score: float
    product_name: str | None = None
    product_group: str | None = None
    image_path: str | None = None
    reason: str
    signals: dict[str, float] = Field(default_factory=dict)
    evidence_article_ids: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    user_id: str | None
    strategy: str
    explanation: str
    recommendations: list[ProductRecommendation]


class HealthResponse(BaseModel):
    status: str
    models_ready: bool
    detail: str


@dataclass
class RecommendationService:
    hybrid: HybridRecommender
    cold_start: ColdStartRecommender
    known_users: set[str]
    metadata: dict[str, dict[str, object]]

    @classmethod
    def load(cls, artifacts_dir: Path, processed_dir: Path) -> "RecommendationService":
        final_dir = artifacts_dir / "final"
        cold_dir = artifacts_dir / "cold_start"
        collaborative = joblib.load(final_dir / "collaborative_svd.joblib")
        content = joblib.load(final_dir / "content_tfidf.joblib")
        cold_start = joblib.load(cold_dir / "cold_start.joblib")
        weight = 0.75
        config_path = artifacts_dir / "hybrid" / "best_hybrid_config.json"
        if config_path.exists():
            weight = float(json.loads(config_path.read_text())["collaborative_weight"])
        hybrid = HybridRecommender(collaborative, content, weight)
        columns = [
            "article_id",
            "prod_name",
            "product_group_name",
            "image_path",
        ]
        items = pd.read_parquet(processed_dir / "items_clean.parquet", columns=columns)
        items["article_id"] = items["article_id"].astype(str)
        metadata = items.set_index("article_id").to_dict(orient="index")
        return cls(hybrid, cold_start, set(collaborative.user_ids_), metadata)

    def recommend(self, payload: RecommendationRequest) -> RecommendationResponse:
        if payload.user_id and payload.user_id in self.known_users:
            components = self.hybrid.recommend_with_components(
                payload.user_id, payload.limit
            )
            ranked = [(entry.article_id, entry.score) for entry in components]
            component_lookup = {entry.article_id: entry for entry in components}
            strategy = "hybrid_personalized"
            explanation = "Blended collaborative and product-content preferences."
        else:
            result = self.cold_start.recommend(
                k=payload.limit,
                age=payload.age,
                club_member_status=payload.club_member_status,
                preferred_product_group=payload.preferred_product_group,
                session_article_ids=payload.session_article_ids,
            )
            ranked, strategy, explanation = (
                result.recommendations,
                result.strategy,
                result.explanation,
            )
            component_lookup = {}
        products = []
        for article_id, score in ranked:
            item = self.metadata.get(str(article_id), {})
            component = component_lookup.get(str(article_id))
            evidence_ids: list[str] = []
            reason = explanation
            if component is not None and payload.user_id is not None:
                evidence = self._closest_history_item(payload.user_id, str(article_id))
                if evidence is not None:
                    evidence_ids = [evidence]
                    evidence_name = _optional(
                        self.metadata.get(evidence, {}).get("prod_name")
                    ) or evidence
                    reason = f"Because you interacted with {evidence_name}."
                signals = {
                    "collaborative": component.collaborative_contribution,
                    "content": component.content_contribution,
                }
            else:
                signals = {strategy: 1.0}
            products.append(
                ProductRecommendation(
                    article_id=str(article_id),
                    score=float(score),
                    product_name=_optional(item.get("prod_name")),
                    product_group=_optional(item.get("product_group_name")),
                    image_path=_optional(item.get("image_path")),
                    reason=reason,
                    signals=signals,
                    evidence_article_ids=evidence_ids,
                )
            )
        return RecommendationResponse(
            user_id=payload.user_id,
            strategy=strategy,
            explanation=explanation,
            recommendations=products,
        )

    def _closest_history_item(self, user_id: str, article_id: str) -> str | None:
        """Find the history item most content-similar to the recommendation."""
        collaborative = self.hybrid.collaborative
        content = self.hybrid.content
        user_index = collaborative.user_to_index_.get(str(user_id))
        target_index = content.item_to_index_.get(str(article_id))
        if user_index is None or target_index is None:
            return None
        eligible = []
        for index in collaborative.seen_[user_index]:
            history_id = str(collaborative.item_ids_[index])
            content_index = content.item_to_index_.get(history_id)
            if content_index is not None:
                eligible.append((history_id, content_index))
        if not eligible:
            return None
        target = content.item_factors_[target_index]
        return max(
            eligible,
            key=lambda pair: float(content.item_factors_[pair[1]] @ target),
        )[0]


def _optional(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def create_app(service: RecommendationService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        app.state.load_error = None
        if app.state.service is None:
            try:
                app.state.service = RecommendationService.load(
                    Path(os.getenv("RECO_NOVA_ARTIFACTS_DIR", "artifacts")),
                    Path(os.getenv("RECO_NOVA_PROCESSED_DIR", "data/processed")),
                )
            except Exception as exc:  # health must remain available for diagnosis
                app.state.load_error = str(exc)
        yield

    api = FastAPI(
        title="Reco-Nova Recommendation API",
        version="1.0.0",
        lifespan=lifespan,
    )

    def ready(request: Request) -> RecommendationService:
        if request.app.state.service is None:
            raise HTTPException(
                status_code=503,
                detail=f"Recommendation models unavailable: {request.app.state.load_error}",
            )
        return request.app.state.service

    @api.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        is_ready = request.app.state.service is not None
        return HealthResponse(
            status="ok" if is_ready else "degraded",
            models_ready=is_ready,
            detail="Models loaded" if is_ready else str(request.app.state.load_error),
        )

    @api.get("/")
    def root() -> dict[str, str]:
        """Discoverability response for users opening the server in a browser."""
        return {
            "name": "Reco-Nova Recommendation API",
            "health": "/health",
            "documentation": "/docs",
        }

    @api.post("/recommend", response_model=RecommendationResponse)
    def recommend(payload: RecommendationRequest, request: Request) -> RecommendationResponse:
        return ready(request).recommend(payload)

    @api.post("/explain", response_model=RecommendationResponse)
    def explain(payload: RecommendationRequest, request: Request) -> RecommendationResponse:
        """Return recommendations with the routing strategy and reason text."""
        return ready(request).recommend(payload)

    return api


app = create_app()
