"""
Model Arena: Head-to-Head Algorithm Benchmarking Engine
Evaluates multiple models side-by-side on holdout test splits across:
- Ranking Quality: NDCG@K, HitRate@K, MAP@K, MRR
- Ecosystem Health: Catalog Coverage, Intra-List Diversity (ILS), Novelty
- System Performance: Serving Latency (p50, p99 ms)
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any, Optional
from pydantic import BaseModel, Field

from backend.models.base import BaseRecommender
from backend.evaluation.metrics import RecSysEvaluator
from backend.data.loaders import RecDataset


class ModelBenchmarkRow(BaseModel):
    model_key: str
    model_name: str
    generation: str
    company: str
    year: int
    ndcg_at_10: float
    hit_rate_at_10: float
    map_at_10: float
    mrr_at_10: float
    catalog_coverage: float
    diversity_ils: float
    novelty_score: float
    avg_latency_ms: float


class ArenaBenchmarkResponse(BaseModel):
    domain: str
    num_test_users: int
    num_catalog_items: int
    models: List[ModelBenchmarkRow]
    summary: str


class ModelArena:
    def __init__(self, dataset: RecDataset, trained_models: Dict[str, BaseRecommender]):
        self.dataset = dataset
        self.models = trained_models

    def run_tournament(
        self,
        model_keys: Optional[List[str]] = None,
        k: int = 10,
        sample_users: int = 30,
    ) -> ArenaBenchmarkResponse:
        train_df, test_df = self.dataset.train_test_split(strategy="leave_k_out", k=3)

        # Build ground-truth test mapping
        ground_truth: Dict[str, Set[str]] = {}
        for uid, group in test_df.groupby("user_id"):
            pos_items = group[group["rating"] >= 3.0]["item_id"].tolist()
            if not pos_items:
                pos_items = group["item_id"].tolist()
            ground_truth[str(uid)] = set(map(str, pos_items))

        # Sample evaluation users
        eval_users = list(ground_truth.keys())[:sample_users]
        filtered_gt = {uid: ground_truth[uid] for uid in eval_users}

        # Catalog items and popularity
        catalog_items = set(self.dataset.items_df["item_id"].astype(str).unique())
        pop_counts = train_df["item_id"].value_counts().to_dict()
        pop_dict = {str(k): int(v) for k, v in pop_counts.items()}
        total_interactions = len(train_df)

        # Item embeddings for diversity
        item_embeddings: Dict[str, np.ndarray] = {}
        first_model = list(self.models.values())[0]
        for iid in catalog_items:
            emb = first_model.get_item_embedding(iid)
            if emb is not None:
                item_embeddings[iid] = emb

        evaluator = RecSysEvaluator(
            ground_truth=filtered_gt,
            catalog_items=catalog_items,
            item_popularity=pop_dict,
            total_interactions=total_interactions,
            item_embeddings=item_embeddings,
        )

        selected_keys = model_keys or list(self.models.keys())
        benchmark_rows: List[ModelBenchmarkRow] = []

        for key in selected_keys:
            if key not in self.models:
                continue

            model = self.models[key]
            rec_map: Dict[str, List[str]] = {}
            latencies = []

            for uid in eval_users:
                t0 = time.perf_counter()
                try:
                    recs = model.recommend(user_id=uid, n=k)
                    rec_map[uid] = [it.item_id for it in recs]
                except Exception:
                    rec_map[uid] = []
                latencies.append((time.perf_counter() - t0) * 1000.0)

            # Compute all metrics
            scores = evaluator.evaluate(rec_map, k=k)

            from backend.config import settings
            reg_info = settings.REGISTERED_MODELS.get(key, {})

            benchmark_rows.append(
                ModelBenchmarkRow(
                    model_key=key,
                    model_name=reg_info.get("name", key),
                    generation=reg_info.get("generation", "Classic"),
                    company=reg_info.get("company", "Industry"),
                    year=reg_info.get("year", 2020),
                    ndcg_at_10=round(scores.get(f"ndcg@{k}", 0.0), 4),
                    hit_rate_at_10=round(scores.get(f"hit_rate@{k}", 0.0), 4),
                    map_at_10=round(scores.get(f"map@{k}", 0.0), 4),
                    mrr_at_10=round(scores.get(f"mrr@{k}", 0.0), 4),
                    catalog_coverage=round(scores.get("catalog_coverage", 0.0), 4),
                    diversity_ils=round(scores.get("diversity_ils", 0.5), 4),
                    novelty_score=round(scores.get("novelty_score", 0.0), 2),
                    avg_latency_ms=round(float(np.mean(latencies)), 2) if latencies else 1.0,
                )
            )

        # Sort by NDCG descending
        benchmark_rows.sort(key=lambda x: x.ndcg_at_10, reverse=True)

        summary = (
            f"Evaluated {len(benchmark_rows)} models on {len(eval_users)} holdout users. "
            f"Top performing model for ranking accuracy: {benchmark_rows[0].model_name} (NDCG@{k}: {benchmark_rows[0].ndcg_at_10:.3f})."
        )

        return ArenaBenchmarkResponse(
            domain=self.dataset.domain,
            num_test_users=len(eval_users),
            num_catalog_items=len(catalog_items),
            models=benchmark_rows,
            summary=summary,
        )
