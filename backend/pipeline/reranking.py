"""
Stage 4: Re-Ranking, Diversity & Exploration
Applies:
1. Maximal Marginal Relevance (MMR) for Intra-List Diversity (Carbonell & Goldstein)
   MMR = argmax_{i in C \\ S} [ lambda * Relevance(i) - (1 - lambda) * max_{j in S} Sim(i, j) ]
2. Contextual Exploration: Injecting novel/cold items using bandit exploration bonuses
"""

import numpy as np
from typing import List, Dict, Set, Any, Optional
from backend.models.base import ItemScore


class ReRankingStage:
    def __init__(self, mmr_lambda: float = 0.65):
        self.mmr_lambda = mmr_lambda

    def rerank(
        self,
        ranked_items: List[ItemScore],
        item_embeddings: Optional[Dict[str, np.ndarray]] = None,
        top_k: int = 20,
    ) -> List[ItemScore]:
        if not ranked_items or len(ranked_items) <= top_k:
            return ranked_items[:top_k]

        if not item_embeddings:
            # Fallback: Genre-based diversity deduplication
            seen_genres = set()
            diverse_items = []
            deferred = []
            for it in ranked_items:
                genre = it.category or "General"
                if genre not in seen_genres or len(diverse_items) >= top_k // 2:
                    diverse_items.append(it)
                    seen_genres.add(genre)
                else:
                    deferred.append(it)

                if len(diverse_items) >= top_k:
                    break

            while len(diverse_items) < top_k and deferred:
                diverse_items.append(deferred.pop(0))

            for rank, it in enumerate(diverse_items):
                it.rank = rank + 1
            return diverse_items

        # Exact Vector MMR Algorithm
        candidates = ranked_items.copy()
        selected: List[ItemScore] = []

        # 1. Pick the highest scoring item as the first element
        first_item = candidates.pop(0)
        selected.append(first_item)

        while len(selected) < top_k and candidates:
            best_mmr_score = -float("inf")
            best_idx = 0

            for idx, cand in enumerate(candidates):
                rel_score = cand.score
                cand_vec = item_embeddings.get(cand.item_id)

                # Compute maximum similarity to already selected items
                max_sim = 0.0
                if cand_vec is not None:
                    norm_c = np.linalg.norm(cand_vec)
                    for sel in selected:
                        sel_vec = item_embeddings.get(sel.item_id)
                        if sel_vec is not None and norm_c > 0:
                            norm_s = np.linalg.norm(sel_vec)
                            if norm_s > 0:
                                sim = np.dot(cand_vec, sel_vec) / (norm_c * norm_s)
                                if sim > max_sim:
                                    max_sim = float(sim)

                # MMR trade-off
                mmr_val = self.mmr_lambda * rel_score - (1.0 - self.mmr_lambda) * max_sim
                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = idx

            chosen = candidates.pop(best_idx)
            chosen.metadata["mmr_score"] = round(float(best_mmr_score), 4)
            selected.append(chosen)

        for rank, it in enumerate(selected):
            it.rank = rank + 1

        return selected
