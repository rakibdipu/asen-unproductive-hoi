"""
Gen 1: Item-Item Collaborative Filtering (Amazon Landmark Algorithm, Sarwar et al. 2001)
Computes cosine similarity between item interaction vectors and predicts user rating
based on weighted ratings of k-nearest items previously rated by the user.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import defaultdict
from backend.models.base import BaseRecommender, ItemScore, ExplainResult


class ItemCFRecommender(BaseRecommender):
    """
    Item-Item Collaborative Filtering with top-K neighbors:
    score(u, i) = sum_{j in rated(u) ∩ N(i)} sim(i, j) * r_{u, j} / sum_{j} |sim(i, j)|
    """

    def __init__(
        self,
        domain: str = "movies",
        k_neighbors: int = 20,
        min_support: int = 2,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="item_cf", domain=domain, config=config)
        self.k_neighbors = k_neighbors
        self.min_support = min_support
        self.similarity_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.user_history: Dict[str, Dict[str, float]] = defaultdict(dict)

    def fit(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None) -> "ItemCFRecommender":
        if items_df is not None:
            self.item_metadata = items_df.set_index("item_id").to_dict(orient="index")

        # Build user-item rating dictionary
        item_users = defaultdict(dict)
        for _, row in interactions_df.iterrows():
            uid = str(row["user_id"])
            iid = str(row["item_id"])
            rating = float(row.get("rating", 1.0))
            self.user_history[uid][iid] = rating
            item_users[iid][uid] = rating

        item_list = list(item_users.keys())

        # Precompute L2 norms for items
        norms = {}
        for iid, u_dict in item_users.items():
            norms[iid] = np.sqrt(sum(r * r for r in u_dict.values()))

        # Compute cosine similarities between items sharing common users
        for i in range(len(item_list)):
            iid_a = item_list[i]
            users_a = item_users[iid_a]
            norm_a = norms[iid_a]
            if norm_a == 0:
                continue

            for j in range(i + 1, len(item_list)):
                iid_b = item_list[j]
                users_b = item_users[iid_b]
                norm_b = norms[iid_b]
                if norm_b == 0:
                    continue

                # Common raters
                common_users = set(users_a.keys()) & set(users_b.keys())
                if len(common_users) < self.min_support:
                    continue

                dot_product = sum(users_a[u] * users_b[u] for u in common_users)
                sim = dot_product / (norm_a * norm_b)

                if sim > 0:
                    self.similarity_matrix[iid_a][iid_b] = float(sim)
                    self.similarity_matrix[iid_b][iid_a] = float(sim)

        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_item_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ItemScore]:
        user_rated = self.user_history.get(user_id, {})
        exclude_set = set(exclude_item_ids or []).union(set(user_rated.keys()))

        if not user_rated:
            return []

        # Predict score for each candidate item
        candidate_scores = defaultdict(float)
        candidate_sim_sums = defaultdict(float)

        for rated_item, rating in user_rated.items():
            neighbors = self.similarity_matrix.get(rated_item, {})
            for neighbor_item, sim in neighbors.items():
                if neighbor_item in exclude_set:
                    continue
                candidate_scores[neighbor_item] += sim * rating
                candidate_sim_sums[neighbor_item] += abs(sim)

        scored_items = []
        for iid, total_score in candidate_scores.items():
            sim_sum = candidate_sim_sums[iid]
            predicted_rating = (total_score / sim_sum) if sim_sum > 0 else 0.0
            scored_items.append((iid, predicted_rating))

        scored_items.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        max_rating = 5.0
        for rank, (iid, score) in enumerate(scored_items[:n]):
            meta = self.item_metadata.get(iid, {})
            norm_score = min(1.0, max(0.0, score / max_rating))
            recommendations.append(
                ItemScore(
                    item_id=iid,
                    score=round(float(norm_score), 4),
                    rank=rank + 1,
                    title=meta.get("title", f"Item {iid}"),
                    category=meta.get("primary_genre", meta.get("genre", meta.get("category", "General"))),
                    thumbnail_url=meta.get("thumbnail_url"),
                    source_model=self.name,
                    metadata={"predicted_rating": round(float(score), 2)},
                )
            )

        return recommendations

    def explain(self, user_id: str, item_id: str) -> ExplainResult:
        user_rated = self.user_history.get(user_id, {})
        neighbors = self.similarity_matrix.get(item_id, {})
        common = [(iid, user_rated[iid], neighbors[iid]) for iid in user_rated if iid in neighbors]
        common.sort(key=lambda x: x[2], reverse=True)

        meta = self.item_metadata.get(item_id, {})
        title = meta.get("title", f"Item {item_id}")

        top_factors = []
        for iid, r, sim in common[:3]:
            r_meta = self.item_metadata.get(iid, {})
            top_factors.append({
                "item_id": iid,
                "title": r_meta.get("title", f"Item {iid}"),
                "your_rating": r,
                "similarity": round(sim, 3)
            })

        nl_text = "Recommended because people with similar tastes also enjoyed this."
        if top_factors:
            first_title = top_factors[0]["title"]
            nl_text = f"Recommended because you loved '{first_title}' (Similarity: {top_factors[0]['similarity']:.0%})."

        return ExplainResult(
            user_id=user_id,
            item_id=item_id,
            item_title=title,
            score=0.85,
            algorithm=self.name,
            top_contributing_interactions=top_factors,
            natural_language_explanation=nl_text,
        )
