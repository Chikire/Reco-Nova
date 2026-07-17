"""Recommendation system public re-exports.

All production implementations live in ``reco_nova.models``.  Import from
there directly; this module exists only for backwards-compatibility.
"""

from reco_nova.models.collaborative import CollaborativeSVD
from reco_nova.models.cold_start import ColdStartRecommender
from reco_nova.models.content import ContentRecommender
from reco_nova.models.hybrid import HybridRecommender
from reco_nova.models.popularity import PopularityRecommender

__all__ = [
    "CollaborativeSVD",
    "ColdStartRecommender",
    "ContentRecommender",
    "HybridRecommender",
    "PopularityRecommender",
]
