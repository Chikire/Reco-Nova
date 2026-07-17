"""FastAPI serving layer for personalized and cold-start recommendations."""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from reco_nova.assistant import AssistantRequest, AssistantResult, ShoppingAssistant
from reco_nova.models import ColdStartRecommender, HybridRecommender


class RecommendationRequest(BaseModel):
    """Known-user ID plus optional anonymous cold-start context."""

    user_id: str | None = Field(default=None, min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    age: float | None = Field(default=None, ge=13, le=120)
    club_member_status: str | None = None
    preferred_product_group: str | None = None
    session_article_ids: list[str] = Field(default_factory=list, max_length=50)
    use_demographics: bool = True


class ProductRecommendation(BaseModel):
    article_id: str
    score: float
    product_name: str | None = None
    product_group: str | None = None
    colour: str | None = None
    description: str | None = None
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
    prices: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.prices is None:
            self.prices = {}

    @classmethod
    def load(cls, artifacts_dir: Path, processed_dir: Path) -> "RecommendationService":
        final_dir = artifacts_dir / "final"
        cold_dir = artifacts_dir / "cold_start"
        # Models are loaded in a background thread (see create_app lifespan).
        # Plain joblib.load is used; mmap_mode provides no benefit for
        # class-hierarchy pickles (scikit-learn / surprise objects).
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
            "product_type_name",
            "product_group_name",
            "detail_desc",
            "item_text",
            "image_path",
        ]
        items = pd.read_parquet(processed_dir / "items_clean.parquet", columns=columns)
        items["article_id"] = items["article_id"].astype(str)
        metadata = items.set_index("article_id").to_dict(orient="index")
        # Compute median price per article from training interactions.
        # Raw values are normalized; multiply by 1000 to approximate USD.
        # Cache result alongside models so startup only reads parquet once.
        prices: dict[str, float] = {}
        prices_cache = artifacts_dir / "prices.json"
        if prices_cache.exists():
            prices = json.loads(prices_cache.read_text())
        else:
            interactions_path = processed_dir / "interactions_train.parquet"
            if interactions_path.exists():
                tx = pd.read_parquet(
                    interactions_path, columns=["article_id", "price"]
                )
                tx["article_id"] = tx["article_id"].astype(str)
                prices = (
                    tx.groupby("article_id")["price"]
                    .median()
                    .mul(1000)
                    .round(0)
                    .astype(int)
                    .to_dict()
                )
                try:
                    prices_cache.write_text(json.dumps(prices))
                except OSError:
                    pass
        return cls(hybrid, cold_start, set(collaborative.user_ids_), metadata, prices)

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
                use_demographics=payload.use_demographics,
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
                    description=_optional(item.get("item_text"))
                    or _optional(item.get("detail_desc")),
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

    def search_catalog(
        self, query: str, limit: int, budget: float | None = None
    ) -> RecommendationResponse | None:
        """Find concrete product types in catalog text, ranked by purchase popularity."""
        stopwords = {
            "i", "me", "my", "want", "need", "find", "show", "give", "recommend",
            "a", "an", "the", "some", "please", "product", "products", "item", "items",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "under", "below", "above", "over", "less", "than", "more", "budget",
            "price", "cheap", "expensive", "affordable", "max", "maximum",
            "up", "to", "within", "for", "with", "or", "and", "in", "at", "of",
            "is", "are", "have", "has", "can", "do", "not", "no",
        }
        tokens = [
            token for token in re.findall(r"[a-z]+", query.lower())
            if token not in stopwords and len(token) > 1
        ]
        if not tokens:
            return None

        def matches_taxonomy(item: dict[str, object]) -> bool:
            taxonomy = " ".join(
                str(item.get(field, ""))
                for field in ("product_type_name", "product_group_name")
            ).lower()
            for token in tokens:
                forms = {token, token[:-1]} if token.endswith("s") and len(token) > 3 else {token}
                if not any(re.search(rf"\b{re.escape(form)}s?\b", taxonomy) for form in forms):
                    return False
            return True

        popularity = getattr(getattr(self.cold_start, "global_", None), "scores_", {})
        candidates = [
            (article_id, float(popularity.get(article_id, 0.0)))
            for article_id, item in self.metadata.items()
            if matches_taxonomy(item)
            and (budget is None or self.prices.get(article_id, 0) <= budget)
        ]
        ranked = [
            (article_id, score)
            for article_id, score in sorted(candidates, key=lambda entry: (-entry[1], entry[0]))[:limit]
        ]
        if not ranked:
            return RecommendationResponse(
                user_id=None,
                strategy="catalog_text_search",
                explanation=f"No catalog product type matches “{query.strip()}”.",
                recommendations=[],
            )
        products = []
        for article_id, score in ranked:
            item = self.metadata[article_id]
            products.append(
                ProductRecommendation(
                    article_id=article_id,
                    score=score,
                    product_name=_optional(item.get("prod_name")),
                    product_group=_optional(item.get("product_group_name")),
                    colour=f"~${self.prices[article_id]}" if article_id in self.prices else None,
                    description=_optional(item.get("item_text"))
                    or _optional(item.get("detail_desc")),
                    image_path=_optional(item.get("image_path")),
                    reason=f"Matches your request for \u201c{query.strip()}\u201d.",
                    signals={"catalog_text_match": 1.0},
                )
            )
        budget_note = f" under ${int(budget)}" if budget is not None else ""
        return RecommendationResponse(
            user_id=None,
            strategy="catalog_text_search",
            explanation=f"Popular catalog products matching \u201c{query.strip()}\u201d{budget_note}.",
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
        import threading
        app.state.service = service
        app.state.load_error = None
        app.state.loading = False
        if app.state.service is None:
            app.state.loading = True

            def _load() -> None:
                try:
                    repo_root = Path(__file__).resolve().parents[2]
                    app.state.service = RecommendationService.load(
                        Path(
                            os.getenv(
                                "RECO_NOVA_ARTIFACTS_DIR",
                                str(repo_root / "artifacts"),
                            )
                        ),
                        Path(
                            os.getenv(
                                "RECO_NOVA_PROCESSED_DIR",
                                str(repo_root / "data" / "processed"),
                            )
                        ),
                    )
                except Exception as exc:
                    app.state.load_error = str(exc)
                finally:
                    app.state.loading = False

            threading.Thread(target=_load, daemon=True).start()
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
        loading = getattr(request.app.state, "loading", False)
        if is_ready:
            detail = "Models loaded"
            status = "ok"
        elif loading:
            detail = "Models loading in background — check back in a moment"
            status = "loading"
        else:
            detail = str(request.app.state.load_error)
            status = "degraded"
        return HealthResponse(
            status=status,
            models_ready=is_ready,
            detail=detail,
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

    @api.post("/assistant/chat", response_model=AssistantResult)
    def assistant_chat(payload: AssistantRequest, request: Request) -> AssistantResult:
        """Convert natural-language shopping intent into grounded recommendations."""
        service = ready(request)
        groups = sorted(
            {
                str(item["product_group_name"])
                for item in service.metadata.values()
                if item.get("product_group_name")
            }
        )

        def recommend(values: dict[str, object]) -> dict[str, object]:
            catalog_query = str(values.pop("catalog_query", ""))
            budget = values.pop("max_budget", None)
            budget_f = float(budget) if budget is not None else None
            limit = int(values.get("limit", 6))
            result = (
                service.search_catalog(catalog_query, limit, budget=budget_f)
                if catalog_query
                else None
            )
            if result is None or not result.recommendations:
                result = service.recommend(
                    RecommendationRequest.model_validate(values)
                )
                if budget_f is not None and service.prices:
                    result.recommendations = [
                        r for r in result.recommendations
                        if service.prices.get(r.article_id, 0) <= budget_f
                    ]
            return result.model_dump()

        return ShoppingAssistant(recommend, groups).chat(payload)

    return api


app = create_app()
