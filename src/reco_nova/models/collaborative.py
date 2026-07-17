"""Collaborative-filtering baseline using truncated matrix factorization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .popularity import PopularityRecommender


class CollaborativeSVD:
    """Factorize an implicit user-item count matrix with randomized SVD.

    Repeated purchases contribute logarithmically scaled confidence. Unknown
    users fall back to the popularity model, which also makes the model usable
    before the dedicated cold-start milestone.
    """

    def __init__(self, n_components: int = 64, random_state: int = 42) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be greater than zero")
        self.n_components = n_components
        self.random_state = random_state

    def fit(
        self,
        interactions: pd.DataFrame,
        recency_half_life_days: float | None = None,
    ) -> "CollaborativeSVD":
        required = {"customer_id", "article_id"}
        if missing := required - set(interactions.columns):
            raise ValueError(f"Missing interaction columns: {sorted(missing)}")
        use_recency = recency_half_life_days is not None and "event_ts" in interactions.columns
        columns = ["customer_id", "article_id", "event_ts"] if use_recency else [
            "customer_id",
            "article_id",
        ]
        frame = interactions[columns].copy()
        frame["customer_id"] = frame["customer_id"].astype(str)
        frame["article_id"] = frame["article_id"].astype(str)
        if frame.empty:
            raise ValueError("Cannot fit on empty interactions")

        self.user_ids_ = np.sort(frame["customer_id"].unique())
        self.item_ids_ = np.sort(frame["article_id"].unique())
        self.user_to_index_ = {value: i for i, value in enumerate(self.user_ids_)}
        user_idx = frame["customer_id"].map(self.user_to_index_).to_numpy()
        item_index = {value: i for i, value in enumerate(self.item_ids_)}
        item_idx = frame["article_id"].map(item_index).to_numpy()
        if use_recency:
            event_ts = pd.to_datetime(frame["event_ts"])
            age_days = (event_ts.max() - event_ts).dt.total_seconds() / 86_400.0
            decay = np.log(2) / recency_half_life_days
            values = np.exp(-decay * age_days.to_numpy(dtype=np.float64)).astype(
                np.float32
            )
        else:
            values = np.ones(len(frame), dtype=np.float32)
        matrix = csr_matrix(
            (values, (user_idx, item_idx)),
            shape=(len(self.user_ids_), len(self.item_ids_)),
        )
        matrix.data = np.log1p(matrix.data)
        self.seen_ = [set(matrix.getrow(i).indices) for i in range(matrix.shape[0])]

        max_components = min(matrix.shape) - 1
        if max_components < 1:
            raise ValueError("At least two users and two items are required for SVD")
        components = min(self.n_components, max_components)
        self.svd_ = TruncatedSVD(
            n_components=components,
            n_iter=7,
            random_state=self.random_state,
        )
        self.user_factors_ = self.svd_.fit_transform(matrix)
        self.item_factors_ = self.svd_.components_.T
        self.popularity_ = PopularityRecommender().fit(frame)
        return self

    def recommend(self, user_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Return top-K unseen items or popularity results for an unknown user."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not hasattr(self, "user_factors_"):
            raise RuntimeError("Model must be fitted before recommendation")
        user_id = str(user_id)
        if user_id not in self.user_to_index_:
            return self.popularity_.recommend(user_id, k)

        index = self.user_to_index_[user_id]
        scores = self.item_factors_ @ self.user_factors_[index]
        if self.seen_[index]:
            scores[list(self.seen_[index])] = -np.inf
        available = len(scores) - len(self.seen_[index])
        limit = min(k, available)
        if limit <= 0:
            return []
        candidates = np.argpartition(-scores, limit - 1)[:limit]
        ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [(str(self.item_ids_[i]), float(scores[i])) for i in ranked]
