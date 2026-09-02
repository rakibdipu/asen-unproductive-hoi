"""
Gen 1: Time-Decayed Popularity Recommendation Model (Baseline)
Popularity score decays exponentially with elapsed time to favor fresh, trending items.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from backend.models.base import BaseRecommender, ItemScore, ExplainResult


class PopularityRecommender(BaseRecommender):
    """
    Time-Decayed Popularity Recommender:
    S(i) = sum_{event in interactions} e^(-decay_rate * (t_now - t_event))
    """

    def __init__(
        self,
        domain: str = "movies",
        decay_half_life_days: float = 14.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="popularity", domain=domain, config=config)
        self.decay_rate = np.log(2) / (decay_half_life_days * 86400.0)
        self.item_scores: Dict[str, float] = {}
        self.item_interaction_counts: Dict[str, int] = {}
        self.sorted_items: List[Tuple[str, float]] = []

    def fit(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None) -> "PopularityRecommender":
        if items_df is not None:
            self.item_metadata = items_df.set_index("item_id").to_dict(orient="index")

        now = interactions_df["timestamp"].max() if "timestamp" in interactions_df.columns else time.time()

        item_weights = {}
        counts = {}

        for _, row in interactions_df.iterrows():
            iid = str(row["item_id"])
            ts = row.get("timestamp", now)
            # Weight by rating/engagement multiplier if available
            weight = float(row.get("rating", 3.0)) / 5.0
            if row.get("liked", 0) == 1:
                weight *= 1.5

            if hasattr(now - ts, "total_seconds"):
                delta_t = max(0.0, float((now - ts).total_seconds()))
            else:
                try:
                    delta_t = max(0.0, float(now) - float(ts))
                except Exception:
                    delta_t = 0.0
            decay = np.exp(-self.decay_rate * delta_t)

            item_weights[iid] = item_weights.get(iid, 0.0) + (decay * weight)
            counts[iid] = counts.get(iid, 0) + 1

        self.item_scores = item_weights
        self.item_interaction_counts = counts
        self.sorted_items = sorted(item_weights.items(), key=lambda x: x[1], reverse=True)
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_item_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ItemScore]:
        exclude_set = set(exclude_item_ids or [])
        recommendations = []

        max_score = self.sorted_items[0][1] if self.sorted_items else 1.0

        for rank, (iid, raw_score) in enumerate(self.sorted_items):
            if iid in exclude_set:
                continue
            meta = self.item_metadata.get(iid, {})
            norm_score = raw_score / max_score if max_score > 0 else 0.0

            recommendations.append(
                ItemScore(
                    item_id=iid,
                    score=float(norm_score),
                    rank=len(recommendations) + 1,
                    title=meta.get("title", f"Item {iid}"),
                    category=meta.get("primary_genre", meta.get("genre", meta.get("category", "General"))),
                    thumbnail_url=meta.get("thumbnail_url"),
                    source_model=self.name,
                    metadata={"popularity_count": self.item_interaction_counts.get(iid, 0)},
                )
            )
            if len(recommendations) >= n:
                break

        return recommendations

    def explain(self, user_id: str, item_id: str) -> ExplainResult:
        meta = self.item_metadata.get(item_id, {})
        title = meta.get("title", f"Item {item_id}")
        count = self.item_interaction_counts.get(item_id, 0)
        score = self.item_scores.get(item_id, 0.0)
        return ExplainResult(
            user_id=user_id,
            item_id=item_id,
            item_title=title,
            score=float(score),
            algorithm=self.name,
            natural_language_explanation=f"Currently trending! Watched or interacted with {count} times recently across the platform.",
            feature_importance={"global_popularity": 0.8, "recency_decay": 0.2},
        )
