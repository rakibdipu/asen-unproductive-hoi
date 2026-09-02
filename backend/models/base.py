"""
Abstract Base Recommender & Data Transfer Objects for Omni-RecSys
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd


class ItemScore(BaseModel):
    item_id: str
    score: float
    rank: int = 0
    title: Optional[str] = None
    category: Optional[str] = None
    thumbnail_url: Optional[str] = None
    artwork_variant_id: Optional[str] = None
    source_model: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExplainResult(BaseModel):
    user_id: str
    item_id: str
    item_title: str
    score: float
    algorithm: str
    top_contributing_interactions: List[Dict[str, Any]] = Field(default_factory=list)
    attention_weights: List[float] = Field(default_factory=list)
    natural_language_explanation: str = ""
    feature_importance: Dict[str, float] = Field(default_factory=dict)


class RecommendationResult(BaseModel):
    user_id: str
    domain: str
    algorithm: str
    total_candidates_evaluated: int
    latency_ms: float
    items: List[ItemScore]
    pipeline_stage_counts: Optional[Dict[str, int]] = None


class BaseRecommender(ABC):
    """
    Abstract Base Class for all recommendation algorithms.
    Guarantees uniform interface across all 6 algorithm generations.
    """

    def __init__(self, name: str, domain: str = "movies", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.domain = domain
        self.config = config or {}
        self.is_fitted: bool = False
        self.item_id_to_idx: Dict[str, int] = {}
        self.idx_to_item_id: Dict[int, str] = {}
        self.user_id_to_idx: Dict[str, int] = {}
        self.idx_to_user_id: Dict[int, str] = {}
        self.item_metadata: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None) -> "BaseRecommender":
        """
        Train or initialize the recommendation model with interaction data.
        Expected columns in interactions_df: ['user_id', 'item_id', 'rating' or 'interaction', 'timestamp']
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_item_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ItemScore]:
        """
        Produce top-N recommendations for a single user.
        """
        pass

    def batch_recommend(
        self,
        user_ids: List[str],
        n: int = 10,
        exclude_item_ids_per_user: Optional[Dict[str, List[str]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[ItemScore]]:
        """
        Default batch recommendation loop. Subclasses can override for optimized batch matrix math.
        """
        results = {}
        for uid in user_ids:
            exclude = exclude_item_ids_per_user.get(uid, []) if exclude_item_ids_per_user else None
            results[uid] = self.recommend(user_id=uid, n=n, exclude_item_ids=exclude, context=context)
        return results

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Return dense representation vector for user if applicable."""
        return None

    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Return dense representation vector for item if applicable."""
        return None

    def explain(self, user_id: str, item_id: str) -> ExplainResult:
        """Default explanation implementation."""
        meta = self.item_metadata.get(item_id, {})
        title = meta.get("title", f"Item {item_id}")
        return ExplainResult(
            user_id=user_id,
            item_id=item_id,
            item_title=title,
            score=0.5,
            algorithm=self.name,
            natural_language_explanation=f"Recommended by {self.name} based on collective interaction patterns.",
        )
