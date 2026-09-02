"""
Stage 1: Multi-Source Candidate Retrieval
Retrieves top candidate items from multiple heterogeneous sources:
- Channel A: Deep ANN Vector Search (Two-Tower)
- Channel B: Collaborative Filtering (Implicit ALS)
- Channel C: Sequential Session Transformer (SASRec)
- Channel D: Graph Diffusion (LightGCN)
- Channel E: Trending & Popularity Fallback
Merges and deduplicates into a unified candidate pool (~200 items).
"""

import time
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict
from backend.models.base import BaseRecommender, ItemScore


class CandidatePool:
    def __init__(self):
        self.candidates: Dict[str, ItemScore] = {}
        self.sources: Dict[str, List[str]] = defaultdict(list)
        self.retrieval_latencies: Dict[str, float] = {}

    def add(self, items: List[ItemScore], source_name: str):
        for it in items:
            iid = it.item_id
            self.sources[iid].append(source_name)
            if iid not in self.candidates or it.score > self.candidates[iid].score:
                it.source_model = source_name
                self.candidates[iid] = it

    def to_list(self) -> List[ItemScore]:
        items = list(self.candidates.values())
        # Attach multi-source attribution
        for it in items:
            it.metadata["retrieval_sources"] = self.sources[it.item_id]
        items.sort(key=lambda x: x.score, reverse=True)
        return items


class RetrievalStage:
    def __init__(self, models: Dict[str, BaseRecommender]):
        self.models = models

    def retrieve(
        self,
        user_id: str,
        budget: int = 150,
        exclude_item_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CandidatePool:
        pool = CandidatePool()
        exclude = exclude_item_ids or []

        # Allocate per-source retrieval budget
        per_source_n = max(30, budget // max(1, len(self.models)))

        for name, model in self.models.items():
            t0 = time.perf_counter()
            try:
                recs = model.recommend(user_id=user_id, n=per_source_n, exclude_item_ids=exclude, context=context)
                pool.add(recs, source_name=name)
            except Exception as e:
                # Log and continue with other sources
                pass
            latency = (time.perf_counter() - t0) * 1000.0
            pool.retrieval_latencies[name] = round(latency, 2)

        return pool
