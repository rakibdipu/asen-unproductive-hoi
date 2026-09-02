"""
Gen 4: DIN (Deep Interest Network, Alibaba, Zhou et al. 2018)
Computes dynamic, target-aware user interest vectors using local attention units:
v_U(T) = sum_{j=1}^H a(e_j, e_T) * e_j
where attention unit a(e_j, e_T) takes [e_j, e_T, e_j - e_T, e_j * e_T] through an MLP.
Powers the Attention Heatmap Explorer in the frontend!
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Set, Dict, Any, Optional
from collections import defaultdict

from backend.models.base import BaseRecommender, ItemScore, ExplainResult


class AttentionUnit(nn.Module):
    """
    Local Activation Attention Unit:
    Takes historical item embedding (e_hist) and candidate item embedding (e_candidate).
    Computes dynamic attention scalar score.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        # Input features: [e_hist, e_cand, e_hist - e_cand, e_hist * e_cand] -> 4 * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, hist_embeds: torch.Tensor, cand_embed: torch.Tensor) -> torch.Tensor:
        """
        hist_embeds: [B, seq_len, emb_dim]
        cand_embed: [B, emb_dim]
        Returns: attention weights [B, seq_len]
        """
        seq_len = hist_embeds.size(1)
        cand_expanded = cand_embed.unsqueeze(1).expand(-1, seq_len, -1)

        diff = hist_embeds - cand_expanded
        prod = hist_embeds * cand_expanded

        # Concat along embedding dimension: [B, seq_len, 4 * emb_dim]
        combined = torch.cat([hist_embeds, cand_expanded, diff, prod], dim=-1)
        scores = self.mlp(combined).squeeze(-1)  # [B, seq_len]
        return torch.softmax(scores, dim=-1)


class PyTorchDIN(nn.Module):
    def __init__(self, num_items: int, emb_dim: int = 32):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.attention = AttentionUnit(emb_dim)

        # Final prediction MLP: [user_interest_vec, cand_embed] -> 2 * emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, hist_ids: torch.Tensor, cand_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hist_ids: [B, seq_len]
        cand_id: [B]
        Returns: (click_probability, attention_weights)
        """
        hist_e = self.item_emb(hist_ids)  # [B, seq_len, emb_dim]
        cand_e = self.item_emb(cand_id)   # [B, emb_dim]

        # Calculate attention weights over history
        weights = self.attention(hist_e, cand_e)  # [B, seq_len]

        # Weighted sum of historical items: [B, emb_dim]
        user_interest = torch.sum(hist_e * weights.unsqueeze(-1), dim=1)

        feat = torch.cat([user_interest, cand_e], dim=1)
        logit = self.classifier(feat).squeeze(-1)
        return torch.sigmoid(logit), weights


class DINRecommender(BaseRecommender):
    """
    Deep Interest Network Recommender with Heatmap Attention Inspection.
    """

    def __init__(
        self,
        domain: str = "movies",
        emb_dim: int = 32,
        max_seq_len: int = 8,
        epochs: int = 15,
        lr: float = 0.005,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="din", domain=domain, config=config)
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.lr = lr

        self.model: Optional[PyTorchDIN] = None
        self.user_history: Dict[str, List[int]] = defaultdict(list)
        self.user_history_iids: Dict[str, List[str]] = defaultdict(list)
        self.user_seen_items: Dict[str, Set[str]] = {}

    def fit(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None) -> "DINRecommender":
        if items_df is not None:
            self.item_metadata = items_df.set_index("item_id").to_dict(orient="index")

        unique_users = interactions_df["user_id"].unique()
        unique_items = interactions_df["item_id"].unique()

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user_id = {idx: uid for idx, uid in enumerate(unique_users)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item_id = {idx: iid for idx, iid in enumerate(unique_items)}

        self.user_seen_items = {uid: set() for uid in unique_users}

        sorted_df = interactions_df.sort_values(["user_id", "timestamp"])
        for _, row in sorted_df.iterrows():
            uid = str(row["user_id"])
            iid = str(row["item_id"])
            self.user_seen_items[uid].add(iid)
            if iid in self.item_id_to_idx:
                # 1-indexed for embedding padding
                self.user_history[uid].append(self.item_id_to_idx[iid] + 1)
                self.user_history_iids[uid].append(iid)

        self.model = PyTorchDIN(num_items=len(unique_items), emb_dim=self.emb_dim)

        hist_samples = []
        cand_samples = []
        labels = []

        for _, row in interactions_df.iterrows():
            uid = str(row["user_id"])
            iid = str(row["item_id"])
            i_idx = self.item_id_to_idx[iid] + 1

            hist = self.user_history[uid][-self.max_seq_len:]
            if len(hist) < self.max_seq_len:
                hist = [0] * (self.max_seq_len - len(hist)) + hist

            hist_samples.append(hist)
            cand_samples.append(i_idx)
            y_val = 1.0 if (row.get("clicked", 1) == 1 or float(row.get("rating", 3.0)) >= 3.5) else 0.0
            labels.append(y_val)

        h_t = torch.tensor(hist_samples, dtype=torch.long)
        c_t = torch.tensor(cand_samples, dtype=torch.long)
        y_t = torch.tensor(labels, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(h_t, c_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for bh, bc, by in loader:
                optimizer.zero_grad()
                preds, _ = self.model(bh, bc)
                loss = criterion(preds, by)
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

        hist = self.user_history[user_id][-self.max_seq_len:]
        if len(hist) < self.max_seq_len:
            hist = [0] * (self.max_seq_len - len(hist)) + hist

        exclude_set = set(exclude_item_ids or []).union(self.user_seen_items.get(user_id, set()))
        candidate_items = [iid for iid in self.item_id_to_idx if iid not in exclude_set]
        if not candidate_items:
            return []

        cand_indices = [self.item_id_to_idx[iid] + 1 for iid in candidate_items]
        num_cands = len(cand_indices)

        bh = torch.tensor([hist] * num_cands, dtype=torch.long)
        bc = torch.tensor(cand_indices, dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            scores, weights = self.model(bh, bc)
            scores = scores.numpy()

        top_indices = np.argsort(scores)[::-1][:n]

        recommendations = []
        for rank, pos in enumerate(top_indices):
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
                    metadata={"din_attention_score": round(score_val, 4)},
                )
            )

        return recommendations

    def explain(self, user_id: str, item_id: str) -> ExplainResult:
        """
        Extracts exact attention weights over the user's historical items relative to this target item!
        """
        meta = self.item_metadata.get(item_id, {})
        title = meta.get("title", f"Item {item_id}")

        if not self.is_fitted or user_id not in self.user_id_to_idx or item_id not in self.item_id_to_idx:
            return super().explain(user_id, item_id)

        hist = self.user_history[user_id][-self.max_seq_len:]
        raw_hist_iids = self.user_history_iids[user_id][-self.max_seq_len:]
        if len(hist) < self.max_seq_len:
            pad_count = self.max_seq_len - len(hist)
            hist = [0] * pad_count + hist
            raw_hist_iids = [""] * pad_count + raw_hist_iids

        bh = torch.tensor([hist], dtype=torch.long)
        bc = torch.tensor([self.item_id_to_idx[item_id] + 1], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            pred, weights = self.model(bh, bc)
            weights_list = weights[0].numpy().tolist()
            pred_score = float(pred[0])

        top_factors = []
        for iid, w in zip(raw_hist_iids, weights_list):
            if not iid:
                continue
            hist_meta = self.item_metadata.get(iid, {})
            top_factors.append({
                "item_id": iid,
                "title": hist_meta.get("title", f"Item {iid}"),
                "attention_weight": round(float(w), 4),
                "genre": hist_meta.get("primary_genre", "General"),
            })

        top_factors.sort(key=lambda x: x["attention_weight"], reverse=True)
        primary = top_factors[0]["title"] if top_factors else "your history"

        return ExplainResult(
            user_id=user_id,
            item_id=item_id,
            item_title=title,
            score=round(pred_score, 4),
            algorithm=self.name,
            top_contributing_interactions=top_factors,
            attention_weights=weights_list,
            natural_language_explanation=f"DIN Target-Attention: You watched '{primary}', which received {top_factors[0]['attention_weight']:.1%} attention weight when scoring candidate '{title}'.",
            feature_importance={"target_activated_attention": 0.7, "candidate_affinity": 0.3},
        )
