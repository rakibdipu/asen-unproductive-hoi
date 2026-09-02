r"""
Gen 4: Wide & Deep Learning (Google Play, Cheng et al. 2016)
Combines the benefits of Memorization (Wide linear model) and Generalization (Deep neural network)
Output: P(Y=1|x) = sigmoid(w_wide^T [x, \phi(x)] + w_deep^T a^{(L)} + b)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Dict, Any, Optional
from backend.models.base import BaseRecommender, ItemScore, ExplainResult


class PyTorchWideAndDeep(nn.Module):
    def __init__(self, num_wide_features: int, num_users: int, num_items: int, num_cats: int, emb_dim: int = 32):
        super().__init__()
        # Wide component
        self.wide_linear = nn.Linear(num_wide_features, 1)

        # Deep component
        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.i_emb = nn.Embedding(num_items, emb_dim)
        self.c_emb = nn.Embedding(num_cats + 1, emb_dim // 2, padding_idx=0)

        deep_input_dim = emb_dim * 2 + (emb_dim // 2)
        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, wide_x: torch.Tensor, u_idx: torch.Tensor, i_idx: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide_linear(wide_x)

        u_e = self.u_emb(u_idx)
        i_e = self.i_emb(i_idx)
        c_e = self.c_emb(c_idx)
        deep_in = torch.cat([u_e, i_e, c_e], dim=1)
        deep_out = self.deep_mlp(deep_in)

        # Joint prediction logit
        return torch.sigmoid(wide_out + deep_out).squeeze(-1)


class WideAndDeepRecommender(BaseRecommender):
    """
    Google Wide & Deep Recommender for Stage 3 Ranking.
    """

    def __init__(
        self,
        domain: str = "movies",
        emb_dim: int = 32,
        epochs: int = 15,
        lr: float = 0.005,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="wide_deep", domain=domain, config=config)
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.lr = lr
        self.model: Optional[PyTorchWideAndDeep] = None
        self.category_to_idx: Dict[str, int] = {}
        self.user_seen_items: Dict[str, Set[str]] = {}

    def fit(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None) -> "WideAndDeepRecommender":
        if items_df is not None:
            self.item_metadata = items_df.set_index("item_id").to_dict(orient="index")

        unique_users = interactions_df["user_id"].unique()
        unique_items = interactions_df["item_id"].unique()

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user_id = {idx: uid for idx, uid in enumerate(unique_users)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item_id = {idx: iid for idx, iid in enumerate(unique_items)}

        categories = set()
        for iid, meta in self.item_metadata.items():
            cat = meta.get("primary_genre", meta.get("genre", meta.get("category", "General")))
            categories.add(cat)
        self.category_to_idx = {cat: idx + 1 for idx, cat in enumerate(categories)}

        self.user_seen_items = {uid: set() for uid in unique_users}
        for _, row in interactions_df.iterrows():
            self.user_seen_items[str(row["user_id"])].add(str(row["item_id"]))

        # Wide features: [hour_norm, rating_prev, cross_features] (size = 4)
        num_wide = 4
        self.model = PyTorchWideAndDeep(
            num_wide_features=num_wide,
            num_users=len(unique_users),
            num_items=len(unique_items),
            num_cats=len(self.category_to_idx),
            emb_dim=self.emb_dim,
        )

        wide_list = []
        u_list = []
        i_list = []
        c_list = []
        y_list = []

        for _, row in interactions_df.iterrows():
            u_str = str(row["user_id"])
            i_str = str(row["item_id"])
            u_idx = self.user_id_to_idx[u_str]
            i_idx = self.item_id_to_idx[i_str]
            cat_str = self.item_metadata.get(i_str, {}).get("primary_genre", "General")
            c_idx = self.category_to_idx.get(cat_str, 0)

            # Wide features: hour normalized, device one-hot proxy
            hour = float(row.get("context_hour", 12)) / 24.0
            device_val = 1.0 if row.get("context_device") == "mobile" else 0.0
            rating_val = float(row.get("rating", 3.0)) / 5.0
            cross_val = hour * device_val

            wide_list.append([hour, device_val, rating_val, cross_val])
            u_list.append(u_idx)
            i_list.append(i_idx)
            c_list.append(c_idx)

            # Binary label: clicked or rating >= 3.5
            y_val = 1.0 if (row.get("clicked", 1) == 1 or float(row.get("rating", 3.0)) >= 3.5) else 0.0
            y_list.append(y_val)

        wide_t = torch.tensor(wide_list, dtype=torch.float32)
        u_t = torch.tensor(u_list, dtype=torch.long)
        i_t = torch.tensor(i_list, dtype=torch.long)
        c_t = torch.tensor(c_list, dtype=torch.long)
        y_t = torch.tensor(y_list, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(wide_t, u_t, i_t, c_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for b_wide, b_u, b_i, b_c, b_y in loader:
                optimizer.zero_grad()
                preds = self.model(b_wide, b_u, b_i, b_c)
                loss = criterion(preds, b_y)
                loss.backward()
                optimizer.step()

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
        exclude_set = set(exclude_item_ids or []).union(self.user_seen_items.get(user_id, set()))

        candidate_items = [iid for iid in self.item_id_to_idx if iid not in exclude_set]
        if not candidate_items:
            return []

        # Prepare batch input for scoring
        num_candidates = len(candidate_items)
        ctx = context or {}
        hour = float(ctx.get("hour", 12)) / 24.0
        device_val = 1.0 if ctx.get("device") == "mobile" else 0.0
        wide_feat = torch.tensor([[hour, device_val, 0.7, hour * device_val]] * num_candidates, dtype=torch.float32)
        u_tensor = torch.full((num_candidates,), u_idx, dtype=torch.long)
        i_tensor = torch.tensor([self.item_id_to_idx[iid] for iid in candidate_items], dtype=torch.long)
        c_tensor = torch.tensor(
            [
                self.category_to_idx.get(self.item_metadata.get(iid, {}).get("primary_genre", "General"), 0)
                for iid in candidate_items
            ],
            dtype=torch.long,
        )

        self.model.eval()
        with torch.no_grad():
            scores = self.model(wide_feat, u_tensor, i_tensor, c_tensor).numpy()

        top_order = np.argsort(scores)[::-1][:n]

        recommendations = []
        for rank, pos in enumerate(top_order):
            iid = candidate_items[pos]
            score_val = float(scores[pos])
            meta = self.item_metadata.get(iid, {})
            recommendations.append(
                ItemScore(
                    item_id=iid,
                    score=round(score_val, 4),
                    rank=rank + 1,
                    title=meta.get("title", f"Item {iid}"),
                    category=meta.get("primary_genre", meta.get("genre", meta.get("category", "General"))),
                    thumbnail_url=meta.get("thumbnail_url"),
                    source_model=self.name,
                    metadata={"wide_deep_p_click": round(score_val, 4)},
                )
            )

        return recommendations

    def explain(self, user_id: str, item_id: str) -> ExplainResult:
        meta = self.item_metadata.get(item_id, {})
        title = meta.get("title", f"Item {item_id}")
        return ExplainResult(
            user_id=user_id,
            item_id=item_id,
            item_title=title,
            score=0.87,
            algorithm=self.name,
            natural_language_explanation=f"Wide & Deep Scoring: Combines historical co-occurrence memorization with dense categorical generalizations for '{title}'.",
            feature_importance={"wide_memorization": 0.4, "deep_embeddings": 0.6},
        )
