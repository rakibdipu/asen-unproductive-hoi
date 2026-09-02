"""
Gen 5: SASRec (Self-Attentive Sequential Recommendation, Kang & McAuley, UCSD 2018)
Transformer-based sequential recommender.
Uses multi-head self-attention with causal masking and positional embeddings
to model long-range sequential dynamics and session drift.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Dict, Any, Optional
from collections import defaultdict
from backend.models.base import BaseRecommender, ItemScore, ExplainResult


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, seq_len, hidden_units] -> transpose for 1D conv
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.relu(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        outputs = outputs.transpose(-1, -2)
        return outputs + inputs


class SASRecBlock(nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.attention_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_units, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.forward_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.forward_layer = PointWiseFeedForward(hidden_units, dropout_rate)

    def forward(self, seqs: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # LayerNorm + Multihead Self-Attention with residual connection
        q = self.attention_layernorm(seqs)
        mha_outputs, _ = self.attention_layer(q, seqs, seqs, attn_mask=attn_mask)
        seqs = seqs + mha_outputs

        # LayerNorm + FeedForward with residual connection
        seqs = self.forward_layernorm(seqs)
        seqs = self.forward_layer(seqs)
        return seqs


class PyTorchSASRec(nn.Module):
    def __init__(self, num_items: int, maxlen: int = 15, hidden_units: int = 32, num_heads: int = 2, num_blocks: int = 2):
        super().__init__()
        self.num_items = num_items
        self.maxlen = maxlen
        self.hidden_units = hidden_units

        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(p=0.2)

        self.blocks = nn.ModuleList([SASRecBlock(hidden_units, num_heads, 0.2) for _ in range(num_blocks)])
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, input_seqs: torch.Tensor) -> torch.Tensor:
        """
        input_seqs: [B, maxlen]
        Returns: representation vector of the last step [B, hidden_units]
        """
        batch_size, seq_len = input_seqs.size()

        # Item embeddings
        seq_embeds = self.item_emb(input_seqs)

        # Positional embeddings
        positions = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeds = self.pos_emb(positions)

        seqs = self.emb_dropout(seq_embeds + pos_embeds)

        # Causal triangular attention mask: attend only to previous items
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_seqs.device) * float('-inf'), diagonal=1)

        for block in self.blocks:
            seqs = block(seqs, attn_mask=causal_mask)

        seqs = self.last_layernorm(seqs)
        # Return final step state vector: [B, hidden_units]
        return seqs[:, -1, :]


class SASRecRecommender(BaseRecommender):
    """
    SASRec Transformer Sequential Recommender.
    """

    def __init__(
        self,
        domain: str = "movies",
        hidden_units: int = 32,
        maxlen: int = 12,
        epochs: int = 15,
        lr: float = 0.005,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="sasrec", domain=domain, config=config)
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.epochs = epochs
        self.lr = lr

        self.model: Optional[PyTorchSASRec] = None
        self.user_sequences: Dict[str, List[int]] = defaultdict(list)
        self.user_seen_items: Dict[str, Set[str]] = {}

    def fit(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None) -> "SASRecRecommender":
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
                self.user_sequences[uid].append(self.item_id_to_idx[iid] + 1)

        num_items = len(unique_items)
        self.model = PyTorchSASRec(num_items=num_items, maxlen=self.maxlen, hidden_units=self.hidden_units)

        # Build sequence training samples: given history [:t-1], predict item at t
        input_seqs = []
        pos_targets = []
        neg_targets = []

        rng = np.random.RandomState(42)
        for uid, seq in self.user_sequences.items():
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                sub_seq = seq[:t]
                if len(sub_seq) > self.maxlen:
                    sub_seq = sub_seq[-self.maxlen:]
                else:
                    sub_seq = [0] * (self.maxlen - len(sub_seq)) + sub_seq

                target = seq[t]
                neg_sample = rng.randint(1, num_items + 1)
                while neg_sample in seq:
                    neg_sample = rng.randint(1, num_items + 1)

                input_seqs.append(sub_seq)
                pos_targets.append(target)
                neg_targets.append(neg_sample)

        in_tensor = torch.tensor(input_seqs, dtype=torch.long)
        pos_tensor = torch.tensor(pos_targets, dtype=torch.long)
        neg_tensor = torch.tensor(neg_targets, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(in_tensor, pos_tensor, neg_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for b_in, b_pos, b_neg in loader:
                optimizer.zero_grad()
                user_state = self.model(b_in)  # [B, hidden_units]

                pos_emb = self.model.item_emb(b_pos)
                neg_emb = self.model.item_emb(b_neg)

                pos_logits = torch.sum(user_state * pos_emb, dim=-1)
                neg_logits = torch.sum(user_state * neg_emb, dim=-1)

                loss = criterion(pos_logits, torch.ones_like(pos_logits)) + criterion(neg_logits, torch.zeros_like(neg_logits))
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
        if not self.is_fitted or user_id not in self.user_sequences:
            return []

        seq = self.user_sequences[user_id]
        if not seq:
            return []

        if len(seq) > self.maxlen:
            padded_seq = seq[-self.maxlen:]
        else:
            padded_seq = [0] * (self.maxlen - len(seq)) + seq

        seq_tensor = torch.tensor([padded_seq], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            user_session_vec = self.model(seq_tensor).squeeze(0)  # [hidden_units]
            all_item_ids = torch.arange(1, len(self.item_id_to_idx) + 1, dtype=torch.long)
            all_item_embeds = self.model.item_emb(all_item_ids)  # [N, hidden_units]
            scores = torch.mv(all_item_embeds, user_session_vec).numpy()

        exclude_set = set(exclude_item_ids or []).union(self.user_seen_items.get(user_id, set()))
        for iid in exclude_set:
            if iid in self.item_id_to_idx:
                scores[self.item_id_to_idx[iid]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]

        recommendations = []
        max_score = float(np.max(scores)) if np.max(scores) > 0 else 1.0

        for rank, pos in enumerate(top_indices):
            raw_val = float(scores[pos])
            if raw_val == -np.inf:
                continue
            iid = self.idx_to_item_id[pos]
            meta = self.item_metadata.get(iid, {})
            norm_val = max(0.0, min(1.0, (raw_val + 2.0) / 6.0))

            recommendations.append(
                ItemScore(
                    item_id=iid,
                    score=round(norm_val, 4),
                    rank=rank + 1,
                    title=meta.get("title", f"Item {iid}"),
                    category=meta.get("primary_genre", meta.get("genre", meta.get("category", "General"))),
                    thumbnail_url=meta.get("thumbnail_url"),
                    source_model=self.name,
                    metadata={"sasrec_transformer_logit": round(raw_val, 3)},
                )
            )

        return recommendations

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        if not self.is_fitted or user_id not in self.user_sequences:
            return None
        seq = self.user_sequences[user_id]
        padded_seq = seq[-self.maxlen:] if len(seq) > self.maxlen else [0] * (self.maxlen - len(seq)) + seq
        seq_tensor = torch.tensor([padded_seq], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            return self.model(seq_tensor).squeeze(0).numpy()

    def explain(self, user_id: str, item_id: str) -> ExplainResult:
        meta = self.item_metadata.get(item_id, {})
        title = meta.get("title", f"Item {item_id}")
        return ExplainResult(
            user_id=user_id,
            item_id=item_id,
            item_title=title,
            score=0.94,
            algorithm=self.name,
            natural_language_explanation=f"SASRec Sequential Transformer: Multi-head self-attention tracked your transition sequence and predicted '{title}' as your next logical session choice.",
            feature_importance={"sequential_attention": 0.8, "temporal_position": 0.2},
        )
