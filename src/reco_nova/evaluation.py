"""Offline top-K evaluation for recommendation models.

Metrics are computed per user and then macro-averaged so highly active users do
not dominate the report. Duplicate purchases are treated as one relevant item.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OfflineMetrics:
    """Macro-averaged implicit-feedback ranking metrics."""

    k: int
    users_evaluated: int
    ndcg_at_k: float
    map_at_k: float
    hit_rate_at_k: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _unique_top_k(items: Iterable[str], k: int) -> list[str]:
    """Keep the first occurrence of each item, up to ``k`` items."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return list(dict.fromkeys(str(item) for item in items))[:k]


def ndcg_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return normalized discounted cumulative gain for one user."""
    truth = {str(item) for item in relevant}
    if not truth:
        return 0.0
    ranked = _unique_top_k(recommended, k)
    dcg = sum(
        1.0 / log2(rank + 2)
        for rank, item in enumerate(ranked)
        if item in truth
    )
    ideal_hits = min(len(truth), k)
    ideal = sum(1.0 / log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal


def average_precision_at_k(
    recommended: Sequence[str], relevant: Iterable[str], k: int
) -> float:
    """Return average precision at K for one user."""
    truth = {str(item) for item in relevant}
    if not truth:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(_unique_top_k(recommended, k), start=1):
        if item in truth:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(truth), k)


def hit_rate_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return one when at least one relevant item appears in the top K."""
    truth = {str(item) for item in relevant}
    if not truth:
        return 0.0
    return float(bool(set(_unique_top_k(recommended, k)) & truth))


def evaluate_rankings(
    recommendations: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, Iterable[str]],
    k: int = 10,
) -> OfflineMetrics:
    """Evaluate users present in ground truth and recommendation output."""
    users = sorted(set(recommendations) & set(ground_truth))
    users = [user for user in users if set(ground_truth[user])]
    if not users:
        return OfflineMetrics(k, 0, 0.0, 0.0, 0.0)

    ndcg = [ndcg_at_k(recommendations[u], ground_truth[u], k) for u in users]
    map_values = [
        average_precision_at_k(recommendations[u], ground_truth[u], k)
        for u in users
    ]
    hits = [hit_rate_at_k(recommendations[u], ground_truth[u], k) for u in users]
    count = len(users)
    return OfflineMetrics(
        k=k,
        users_evaluated=count,
        ndcg_at_k=sum(ndcg) / count,
        map_at_k=sum(map_values) / count,
        hit_rate_at_k=sum(hits) / count,
    )


def catalog_coverage(
    recommendations: Mapping[str, Sequence[str]], catalog_size: int, k: int
) -> float:
    """Return the fraction of the eligible catalog exposed in top-K lists."""
    if catalog_size <= 0:
        raise ValueError("catalog_size must be greater than zero")
    exposed = {
        item
        for ranked in recommendations.values()
        for item in _unique_top_k(ranked, k)
    }
    return len(exposed) / catalog_size


def bootstrap_metric_intervals(
    recommendations: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, Iterable[str]],
    k: int = 10,
    samples: int = 1_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Estimate macro-metric confidence intervals by resampling users."""
    if samples <= 0:
        raise ValueError("samples must be greater than zero")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    users = sorted(set(recommendations) & set(ground_truth))
    users = [user for user in users if set(ground_truth[user])]
    if not users:
        return {
            "ndcg_at_k": (0.0, 0.0),
            "map_at_k": (0.0, 0.0),
            "hit_rate_at_k": (0.0, 0.0),
        }
    values = np.array(
        [
            [
                ndcg_at_k(recommendations[user], ground_truth[user], k),
                average_precision_at_k(
                    recommendations[user], ground_truth[user], k
                ),
                hit_rate_at_k(recommendations[user], ground_truth[user], k),
            ]
            for user in users
        ],
        dtype=float,
    )
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(users), size=(samples, len(users)))
    bootstrap_means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    bounds = np.quantile(bootstrap_means, [tail, 1.0 - tail], axis=0)
    names = ("ndcg_at_k", "map_at_k", "hit_rate_at_k")
    return {
        name: (float(bounds[0, i]), float(bounds[1, i]))
        for i, name in enumerate(names)
    }
