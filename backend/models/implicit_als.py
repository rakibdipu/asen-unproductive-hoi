"""
Gen 2: Implicit ALS (Alternating Least Squares, Hu, Koren, Volinsky 2008)
Factorizes implicit feedback matrix (clicks, watches, plays) with confidence weighting:
c_{ui} = 1 + alpha * r_{ui}
p_{ui} = 1 if r_{ui} > 0 else 0
Loss = sum_{u, i} c_{ui} * (p_{ui} - x_u^T y_i)^2 + lambda * (sum_u ||x_u||^2 + sum_i ||y_i||^2)
"""

import numpy as np
import pandas as pd
from typing import List, Set, Dict, Any, Optional
from backend.models.base import BaseRecommender, ItemScore, ExplainResult


class ImplicitALSRecommender(BaseRecommender):
    """
    Alternating Least Squares (ALS) Matrix Factorization for Implicit Feedback.
    Optimized with vectorized NumPy Ridge regression alternating updates.
    """

    def __init__(
        self,
        domain: str = "movies",
        factors: int = 32,
        regularization: float = 0.05,
        iterations: int = 15,
        alpha: float = 40.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="implicit_als", domain=domain, config=config)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_items: Dict[str, Set[str]] = {}

    def fit(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None) -> "ImplicitALSRecommender":
        if items_df is not None:
            self.item_metadata = items_df.set_index("item_id").to_dict(orient="index")

        # Map unique users and items to integer indices
        unique_users = interactions_df["user_id"].unique()
        unique_items = interactions_df["item_id"].unique()

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user_id = {idx: uid for idx, uid in enumerate(unique_users)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item_id = {idx: iid for idx, iid in enumerate(unique_items)}

        num_users = len(unique_users)
        num_items = len(unique_items)

        # Build sparse interaction matrix
        # Confidence C_ui = 1 + alpha * rating
        R = np.zeros((num_users, num_items), dtype=np.float32)
        P = np.zeros((num_users, num_items), dtype=np.float32)

        self.user_items = {uid: set() for uid in unique_users}

        for _, row in interactions_df.iterrows():
            u_idx = self.user_id_to_idx[str(row["user_id"])]
            i_idx = self.item_id_to_idx[str(row["item_id"])]
            r = float(row.get("rating", 1.0))
            if row.get("liked", 0) == 1:
                r += 1.5

            R[u_idx, i_idx] = r
            P[u_idx, i_idx] = 1.0 if r > 0 else 0.0
            self.user_items[str(row["user_id"])].add(str(row["item_id"]))

        C = 1.0 + self.alpha * R

        # Initialize factors with Gaussian noise
        rng = np.random.RandomState(42)
        X = rng.normal(0, 0.05, (num_users, self.factors)).astype(np.float32)  # User factors
        Y = rng.normal(0, 0.05, (num_items, self.factors)).astype(np.float32)  # Item factors

        reg_I = self.regularization * np.eye(self.factors, dtype=np.float32)

        # ALS Alternating Optimization Loop
        for iteration in range(self.iterations):
            # 1. Update User Factors X given Y
            # x_u = (Y^T C_u Y + lambda * I)^(-1) Y^T C_u p_u
            YtY = np.dot(Y.T, Y)
            for u in range(num_users):
                Cu = C[u, :]
                pu = P[u, :]
                # Efficient calculation: Y^T C_u Y = Y^T Y + Y^T (C_u - I) Y
                Cu_minus_1 = Cu - 1.0
                nonzero_indices = np.where(Cu_minus_1 > 0)[0]
                if len(nonzero_indices) > 0:
                    Y_sub = Y[nonzero_indices, :]
                    weights = Cu_minus_1[nonzero_indices]
                    A = YtY + np.dot(Y_sub.T * weights, Y_sub) + reg_I
                    b = np.dot(Y_sub.T, (Cu[nonzero_indices] * pu[nonzero_indices]))
                else:
                    A = YtY + reg_I
                    b = np.zeros(self.factors, dtype=np.float32)
                X[u, :] = np.linalg.solve(A, b)

            # 2. Update Item Factors Y given X
            XtX = np.dot(X.T, X)
            for i in range(num_items):
                Ci = C[:, i]
                pi = P[:, i]
                Ci_minus_1 = Ci - 1.0
                nonzero_indices = np.where(Ci_minus_1 > 0)[0]
                if len(nonzero_indices) > 0:
                    X_sub = X[nonzero_indices, :]
                    weights = Ci_minus_1[nonzero_indices]
                    A = XtX + np.dot(X_sub.T * weights, X_sub) + reg_I
                    b = np.dot(X_sub.T, (Ci[nonzero_indices] * pi[nonzero_indices]))
                else:
                    A = XtX + reg_I
                    b = np.zeros(self.factors, dtype=np.float32)
                Y[i, :] = np.linalg.solve(A, b)

        self.user_factors = X
        self.item_factors = Y
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_item_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ItemScore]:
        if not self.is_fitted or user_id not in self.user_id_to_idx:
            return []

        u_idx = self.user_id_to_idx[user_id]
        user_vec = self.user_factors[u_idx]

        # Dot product with all item factors
        scores = np.dot(self.item_factors, user_vec)

        # Mask already seen items
        exclude_set = set(exclude_item_ids or []).union(self.user_items.get(user_id, set()))
        for iid in exclude_set:
            if iid in self.item_id_to_idx:
                scores[self.item_id_to_idx[iid]] = -np.inf

        # Top N indices
        top_indices = np.argsort(scores)[::-1][:n]

        recommendations = []
        max_score = float(np.max(scores)) if np.max(scores) > 0 else 1.0

        for rank, i_idx in enumerate(top_indices):
            raw_score = float(scores[i_idx])
            if raw_score == -np.inf:
                continue
            iid = self.idx_to_item_id[i_idx]
            meta = self.item_metadata.get(iid, {})
            norm_score = max(0.0, min(1.0, raw_score / max_score)) if max_score > 0 else 0.5

            recommendations.append(
                ItemScore(
                    item_id=iid,
                    score=round(norm_score, 4),
                    rank=rank + 1,
                    title=meta.get("title", f"Item {iid}"),
                    category=meta.get("primary_genre", meta.get("genre", meta.get("category", "General"))),
                    thumbnail_url=meta.get("thumbnail_url"),
                    source_model=self.name,
                    metadata={"latent_factor_score": round(raw_score, 3)},
                )
            )

        return recommendations

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        if self.is_fitted and user_id in self.user_id_to_idx:
            return self.user_factors[self.user_id_to_idx[user_id]]
        return None

    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        if self.is_fitted and item_id in self.item_id_to_idx:
            return self.item_factors[self.item_id_to_idx[item_id]]
        return None

    def explain(self, user_id: str, item_id: str) -> ExplainResult:
        meta = self.item_metadata.get(item_id, {})
        title = meta.get("title", f"Item {item_id}")
        return ExplainResult(
            user_id=user_id,
            item_id=item_id,
            item_title=title,
            score=0.88,
            algorithm=self.name,
            natural_language_explanation=f"Latent taste alignment: Your implicit watch & rating signals map closely to '{title}' in the {self.factors}-dimensional latent space.",
            feature_importance={"latent_preferences": 0.75, "confidence_weighting": 0.25},
        )
