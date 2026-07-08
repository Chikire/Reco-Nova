"""Offline recommendation metrics."""

from dataclasses import dataclass


@dataclass
class OfflineMetrics:
    """Standard ranking metrics for held-out evaluation."""

    ndcg: float = 0.0
    map_at_k: float = 0.0
    hit_rate: float = 0.0
