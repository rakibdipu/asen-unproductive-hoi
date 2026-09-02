"""
Stage 3: Heavy Scoring & Multi-Task Ranking
Takes filtered candidates and executes heavy deep neural scoring (MMoE / DeepFM / Wide&Deep).
Evaluates multi-objective utility functions balancing CTR, Watch Time, and Rating.
"""

from typing import List, Dict, Any, Optional
from backend.models.base import BaseRecommender, ItemScore


class RankingStage:
    def __init__(self, ranker_model: BaseRecommender):
        self.ranker_model = ranker_model

    def rank(
        self,
        user_id: str,
        candidates: List[ItemScore],
        budget: int = 50,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ItemScore]:
        if not candidates:
            return []

        # Get detailed scores from the ranker model
        candidate_ids = [it.item_id for it in candidates]
        try:
            detailed_scores = self.ranker_model.recommend(
                user_id=user_id,
                n=len(candidate_ids),
                exclude_item_ids=[],
                context=context,
            )
            score_map = {it.item_id: it for it in detailed_scores}
        except Exception:
            score_map = {}

        # Re-score candidates combining retrieval relevance and deep ranking
        scored_candidates = []
        for it in candidates:
            iid = it.item_id
            if iid in score_map:
                ranked_it = score_map[iid]
                # Combined utility: 0.3 * retrieval_score + 0.7 * deep_ranking_score
                blended_score = 0.3 * it.score + 0.7 * ranked_it.score
                it.score = round(float(blended_score), 4)
                it.metadata.update(ranked_it.metadata)
            scored_candidates.append(it)

        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        for rank, it in enumerate(scored_candidates[:budget]):
            it.rank = rank + 1

        return scored_candidates[:budget]
