"""Content-based recommendations from product text metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .popularity import PopularityRecommender


class ContentRecommender:
    """Recommend products similar to a user's purchased-product profile.

    Product metadata is represented with TF-IDF and compressed using randomized
    SVD. User profiles are confidence-weighted centroids of product factors.
    Keeping this model local and deterministic makes it a useful comparison for
    later sentence-transformer and multimodal embeddings.
    """

    def __init__(
        self,
        n_components: int = 64,
        max_features: int = 20_000,
        random_state: int = 42,
    ) -> None:
        if n_components <= 0 or max_features <= 0:
            raise ValueError("n_components and max_features must be positive")
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        candidate_item_ids: set[str] | None = None,
    ) -> "ContentRecommender":
        required_interactions = {"customer_id", "article_id"}
        required_items = {"article_id", "item_text"}
        if missing := required_interactions - set(interactions.columns):
            raise ValueError(f"Missing interaction columns: {sorted(missing)}")
        if missing := required_items - set(items.columns):
            raise ValueError(f"Missing item columns: {sorted(missing)}")

        catalog = items[["article_id", "item_text"]].dropna(subset=["article_id"])
        catalog = catalog.assign(
            article_id=catalog["article_id"].astype(str),
            item_text=catalog["item_text"].fillna("").astype(str),
        ).drop_duplicates("article_id")
        if candidate_item_ids is not None:
            eligible = {str(item) for item in candidate_item_ids}
            catalog = catalog[catalog["article_id"].isin(eligible)]
        if len(catalog) < 2:
            raise ValueError("At least two catalog items are required")

        frame = interactions[["customer_id", "article_id"]].dropna().astype(str)
        training_item_ids = set(frame["article_id"])

        self.item_ids_ = catalog["article_id"].to_numpy()
        self.item_to_index_ = {item: i for i, item in enumerate(self.item_ids_)}
        self.training_item_ids_ = {
            item for item in training_item_ids if item in self.item_to_index_
        }
        self.fresh_item_ids_ = set(self.item_to_index_) - self.training_item_ids_
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
        )
        features = self.vectorizer_.fit_transform(catalog["item_text"])
        max_components = min(features.shape) - 1
        if max_components < 1:
            raise ValueError("Item text must contain at least two distinct terms")
        components = min(self.n_components, max_components)
        self.svd_ = TruncatedSVD(
            n_components=components, n_iter=7, random_state=self.random_state
        )
        self.item_factors_ = normalize(
            self.svd_.fit_transform(features), norm="l2", axis=1
        ).astype(np.float32)

        frame = frame[frame["article_id"].isin(self.item_to_index_)]
        counts = frame.groupby(["customer_id", "article_id"]).size().reset_index(name="n")
        self.user_ids_ = np.sort(counts["customer_id"].unique())
        self.user_to_index_ = {user: i for i, user in enumerate(self.user_ids_)}
        rows = counts["customer_id"].map(self.user_to_index_).to_numpy()
        cols = counts["article_id"].map(self.item_to_index_).to_numpy()
        weights = np.log1p(counts["n"].to_numpy(dtype=np.float32))
        history = csr_matrix(
            (weights, (rows, cols)),
            shape=(len(self.user_ids_), len(self.item_ids_)),
        )
        self.seen_ = [set(history.getrow(i).indices) for i in range(history.shape[0])]
        self.user_factors_ = normalize(history @ self.item_factors_, axis=1).astype(
            np.float32
        )
        self.popularity_ = PopularityRecommender().fit(frame)
        return self

    def _score_user_items(self, user_id: str) -> tuple[np.ndarray, int] | None:
        """Return item scores and user index for a known user."""
        user_id = str(user_id)
        if user_id not in self.user_to_index_:
            return None
        index = self.user_to_index_[user_id]
        scores = self.item_factors_ @ self.user_factors_[index]
        scores = np.asarray(scores).reshape(-1)
        if self.seen_[index]:
            scores[list(self.seen_[index])] = -np.inf
        return scores, index

    def _top_k_from_scores(
        self,
        scores: np.ndarray,
        k: int,
        allowed_indices: set[int] | None = None,
    ) -> list[tuple[str, float]]:
        """Convert score vector to ranked item-score tuples."""
        if allowed_indices is not None:
            mask = np.ones(len(scores), dtype=bool)
            if allowed_indices:
                allowed = np.fromiter(allowed_indices, dtype=int)
                mask[allowed] = False
            scores[mask] = -np.inf
        finite = np.isfinite(scores)
        available = int(np.sum(finite))
        limit = min(k, available)
        if limit <= 0:
            return []
        candidates = np.argpartition(-scores, limit - 1)[:limit]
        ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [(str(self.item_ids_[i]), float(scores[i])) for i in ranked]

    def recommend(self, user_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Return products nearest to the user's content preference centroid."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not hasattr(self, "user_factors_"):
            raise RuntimeError("Model must be fitted before recommendation")
        scored = self._score_user_items(user_id)
        if scored is None:
            return self.popularity_.recommend(user_id, k)
        scores, _ = scored
        return self._top_k_from_scores(scores.copy(), k)

    def recommend_fresh_for_user(
        self, user_id: str, k: int = 10
    ) -> list[tuple[str, float]]:
        """Return top-K fresh-catalog items for a known user profile."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not hasattr(self, "user_factors_"):
            raise RuntimeError("Model must be fitted before recommendation")
        scored = self._score_user_items(user_id)
        if scored is None or not self.fresh_item_ids_:
            return []
        scores, _ = scored
        fresh_indices = {
            self.item_to_index_[item]
            for item in self.fresh_item_ids_
            if item in self.item_to_index_
        }
        return self._top_k_from_scores(scores.copy(), k, allowed_indices=fresh_indices)

    def recommend_from_items(
        self, article_ids: list[str], k: int = 10
    ) -> list[tuple[str, float]]:
        """Recommend from anonymous session items without requiring a user ID."""
        indices = [
            self.item_to_index_[str(item)]
            for item in article_ids
            if str(item) in self.item_to_index_
        ]
        if not indices:
            return []
        profile = normalize(self.item_factors_[indices].mean(axis=0).reshape(1, -1))[0]
        scores = np.asarray(self.item_factors_ @ profile).reshape(-1)
        scores[indices] = -np.inf
        limit = min(k, len(scores) - len(set(indices)))
        candidates = np.argpartition(-scores, limit - 1)[:limit]
        ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [(str(self.item_ids_[i]), float(scores[i])) for i in ranked]
