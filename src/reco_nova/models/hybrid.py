"""Score-normalized hybrid ranking ensemble."""

from __future__ import annotations

from typing import Protocol


class Recommender(Protocol):
    def recommend(self, user_id: str, k: int = 10) -> list[tuple[str, float]]: ...


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """Normalize candidate scores to [0, 1] within one user request."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {item: 1.0 for item in scores}
    scale = high - low
    return {item: (score - low) / scale for item, score in scores.items()}


class HybridRecommender:
    """Blend collaborative and content candidates after per-user normalization."""

    def __init__(
        self,
        collaborative: Recommender,
        content: Recommender,
        collaborative_weight: float = 0.5,
        candidate_multiplier: int = 10,
    ) -> None:
        if not 0.0 <= collaborative_weight <= 1.0:
            raise ValueError("collaborative_weight must be between zero and one")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        self.collaborative = collaborative
        self.content = content
        self.collaborative_weight = collaborative_weight
        self.candidate_multiplier = candidate_multiplier

    def recommend(self, user_id: str, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        candidate_k = k * self.candidate_multiplier
        collaborative = dict(self.collaborative.recommend(user_id, candidate_k))
        content = dict(self.content.recommend(user_id, candidate_k))
        collaborative = _minmax(collaborative)
        content = _minmax(content)
        candidates = collaborative.keys() | content.keys()
        weight = self.collaborative_weight
        scores = {
            item: weight * collaborative.get(item, 0.0)
            + (1.0 - weight) * content.get(item, 0.0)
            for item in candidates
        }
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k]
