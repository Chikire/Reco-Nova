"""Popularity baseline and cold-start fallback."""

from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """Rank items by interaction count with deterministic tie-breaking."""

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        required = {"customer_id", "article_id"}
        if missing := required - set(interactions.columns):
            raise ValueError(f"Missing interaction columns: {sorted(missing)}")
        frame = interactions.assign(
            customer_id=interactions["customer_id"].astype(str),
            article_id=interactions["article_id"].astype(str),
        )
        counts = frame.groupby("article_id").size().rename("score").reset_index()
        counts = counts.sort_values(
            ["score", "article_id"], ascending=[False, True]
        )
        self.item_ids_ = counts["article_id"].tolist()
        self.scores_ = dict(zip(counts["article_id"], counts["score"].astype(float)))
        self.seen_ = frame.groupby("customer_id")["article_id"].agg(set).to_dict()
        return self

    def recommend(self, user_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Recommend unseen popular items; unknown users receive global popularity."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not hasattr(self, "item_ids_"):
            raise RuntimeError("Model must be fitted before recommendation")
        seen = self.seen_.get(str(user_id), set())
        output = []
        for item_id in self.item_ids_:
            if item_id not in seen:
                output.append((item_id, self.scores_[item_id]))
                if len(output) == k:
                    break
        return output
