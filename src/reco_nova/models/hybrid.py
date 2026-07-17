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
        fresh_item_ids: set[str] | None = None,
        min_fresh_in_top_k: int = 0,
    ) -> None:
        if not 0.0 <= collaborative_weight <= 1.0:
            raise ValueError("collaborative_weight must be between zero and one")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        if min_fresh_in_top_k < 0:
            raise ValueError("min_fresh_in_top_k must be non-negative")
        self.collaborative = collaborative
        self.content = content
        self.collaborative_weight = collaborative_weight
        self.candidate_multiplier = candidate_multiplier
        self.fresh_item_ids = {str(item) for item in fresh_item_ids or set()}
        self.min_fresh_in_top_k = min_fresh_in_top_k

    def _fresh_content_candidates(
        self, user_id: str, candidate_k: int
    ) -> list[tuple[str, float]]:
        """Return fresh-only content candidates when the content model supports it."""
        if not self.fresh_item_ids:
            return []
        if hasattr(self.content, "recommend_fresh_for_user"):
            output = self.content.recommend_fresh_for_user(user_id, candidate_k)
            return [(str(item), float(score)) for item, score in output]
        output = self.content.recommend(user_id, candidate_k)
        return [
            (str(item), float(score))
            for item, score in output
            if str(item) in self.fresh_item_ids
        ]

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
        output = ranked[:k]
        required_fresh = min(self.min_fresh_in_top_k, k)
        if required_fresh <= 0 or not self.fresh_item_ids:
            return output

        fresh_in_output = [item for item, _ in output if item in self.fresh_item_ids]
        if len(fresh_in_output) >= required_fresh:
            return output

        needed = required_fresh - len(fresh_in_output)
        fresh_ranked = self._fresh_content_candidates(user_id, candidate_k)
        fresh_ranked = sorted(fresh_ranked, key=lambda pair: (-pair[1], pair[0]))
        additions = []
        existing = {item for item, _ in output}
        for item, score in fresh_ranked:
            if item in existing:
                continue
            additions.append((item, score))
            if len(additions) == needed:
                break
        if not additions:
            return output

        remove_count = len(additions)
        pruned = []
        removed = 0
        for item, score in reversed(output):
            if removed < remove_count and item not in self.fresh_item_ids:
                removed += 1
                continue
            pruned.append((item, score))
        output = list(reversed(pruned))
        output.extend(additions)
        output = sorted(output, key=lambda pair: (-pair[1], pair[0]))
        return output[:k]
