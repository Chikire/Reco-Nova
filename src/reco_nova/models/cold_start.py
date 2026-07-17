"""Explainable fallback strategies for users without interaction history."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .content import ContentRecommender
from .popularity import PopularityRecommender


@dataclass(frozen=True)
class ColdStartResult:
    strategy: str
    recommendations: list[tuple[str, float]]
    explanation: str


def age_band(age: float | None) -> str:
    if age is None or pd.isna(age):
        return "unknown"
    if age < 25:
        return "16-24"
    if age < 35:
        return "25-34"
    if age < 50:
        return "35-49"
    return "50+"


class ColdStartRecommender:
    """Use session, demographic, category, then global popularity fallbacks."""

    def __init__(self, min_segment_events: int = 50) -> None:
        self.min_segment_events = min_segment_events

    def fit(
        self,
        interactions: pd.DataFrame,
        customers: pd.DataFrame,
        items: pd.DataFrame,
        content: ContentRecommender | None = None,
    ) -> "ColdStartRecommender":
        events = interactions[["customer_id", "article_id"]].dropna().astype(str)
        self.global_ = PopularityRecommender().fit(events)
        self.content_ = content
        customer_columns = ["customer_id", "age", "club_member_status"]
        customer_data = customers[customer_columns].copy()
        customer_data["customer_id"] = customer_data["customer_id"].astype(str)
        customer_data["age_band"] = customer_data["age"].map(age_band)
        customer_data["club_member_status"] = (
            customer_data["club_member_status"].fillna("none").astype(str).str.lower()
        )
        joined = events.merge(customer_data, on="customer_id", how="left")
        joined["age_band"] = joined["age_band"].fillna("unknown")
        joined["club_member_status"] = joined["club_member_status"].fillna("none")
        self.segment_rankings_: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for key, frame in joined.groupby(["age_band", "club_member_status"]):
            if len(frame) >= self.min_segment_events:
                counts = frame.groupby("article_id").size().sort_values(ascending=False)
                self.segment_rankings_[key] = [
                    (str(item), float(score)) for item, score in counts.items()
                ]

        catalog = items[["article_id", "product_group_name"]].copy()
        catalog["article_id"] = catalog["article_id"].astype(str)
        catalog["product_group_name"] = (
            catalog["product_group_name"].fillna("").astype(str).str.lower()
        )
        counts = events.groupby("article_id").size().rename("score").reset_index()
        counts = counts.merge(catalog, on="article_id", how="left")
        self.category_rankings_ = {
            category: [
                (str(row.article_id), float(row.score))
                for row in frame.sort_values(
                    ["score", "article_id"], ascending=[False, True]
                ).itertuples()
            ]
            for category, frame in counts.groupby("product_group_name")
            if category
        }
        return self

    def recommend(
        self,
        k: int = 10,
        age: float | None = None,
        club_member_status: str | None = None,
        preferred_product_group: str | None = None,
        session_article_ids: list[str] | None = None,
        use_demographics: bool = True,
    ) -> ColdStartResult:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if session_article_ids and self.content_ is not None:
            output = self.content_.recommend_from_items(session_article_ids, k)
            if output:
                return ColdStartResult(
                    "session_content", output, "Based on products viewed this session."
                )
        if use_demographics:
            key = (age_band(age), str(club_member_status or "none").lower())
            if key in self.segment_rankings_:
                return ColdStartResult(
                    "demographic_popularity",
                    self.segment_rankings_[key][:k],
                    f"Popular with shoppers in age band {key[0]} and membership {key[1]}.",
                )
        if preferred_product_group:
            category = preferred_product_group.strip().lower()
            if category in self.category_rankings_:
                return ColdStartResult(
                    "category_popularity",
                    self.category_rankings_[category][:k],
                    f"Popular products in {category}.",
                )
        return ColdStartResult(
            "global_popularity",
            self.global_.recommend("__new_user__", k),
            "Popular products across all shoppers.",
        )
