"""
Omni-RecSys Full Pipeline Orchestrator
Coordinates all 5 production stages:
1. Candidate Retrieval (ANN + ALS + SASRec + LightGCN + Popularity) [~200 items]
2. Business Filtering & Deduplication [~100 items]
3. Deep Multi-Task Ranking (MMoE / DeepFM) [~50 items]
4. Re-Ranking & MMR Intra-List Diversity [~20 items]
5. Slate Generation & LinUCB Artwork Personalization [5 Rows, 20 items]
Provides live funnel metrics (counts and latencies) for the animated Funnel Chart in the UI.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from backend.models.base import BaseRecommender, ItemScore
from backend.pipeline.retrieval import RetrievalStage
from backend.pipeline.filtering import FilteringStage
from backend.pipeline.ranking import RankingStage
from backend.pipeline.reranking import ReRankingStage
from backend.pipeline.slate import SlateStage, SlatePage


class StageMetric(BaseModel):
    stage_name: str
    stage_number: int
    input_count: int
    output_count: int
    latency_ms: float
    description: str


class PipelineResult(BaseModel):
    user_id: str
    domain: str
    total_latency_ms: float
    stage_metrics: List[StageMetric]
    slate_page: SlatePage
    final_items: List[ItemScore]


class ProductionPipeline:
    def __init__(
        self,
        retrieval_models: Dict[str, BaseRecommender],
        ranking_model: BaseRecommender,
        artwork_bandit: Optional[Any] = None,
    ):
        self.retrieval_stage = RetrievalStage(retrieval_models)
        self.filtering_stage = FilteringStage()
        self.ranking_stage = RankingStage(ranking_model)
        self.reranking_stage = ReRankingStage(mmr_lambda=0.65)
        self.slate_stage = SlateStage(artwork_bandit)

    def execute(
        self,
        user_id: str,
        domain: str = "movies",
        catalog_size: int = 150,
        consumed_items: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        user_genre_vector: Optional[np.ndarray] = None,
        item_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> PipelineResult:
        start_pipeline = time.perf_counter()
        stage_metrics: List[StageMetric] = []
        ctx = context or {}

        # -------------------------------------------------------------
        # STAGE 1: CANDIDATE GENERATION (RETRIEVAL)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        candidate_pool = self.retrieval_stage.retrieve(
            user_id=user_id,
            budget=200,
            exclude_item_ids=consumed_items,
            context=ctx,
        )
        candidates = candidate_pool.to_list()
        t1 = time.perf_counter()
        stage_metrics.append(
            StageMetric(
                stage_name="Candidate Retrieval",
                stage_number=1,
                input_count=catalog_size,
                output_count=len(candidates),
                latency_ms=round((t1 - t0) * 1000.0, 2),
                description="Multi-source retrieval via Two-Tower ANN, Implicit ALS, SASRec & LightGCN",
            )
        )

        # -------------------------------------------------------------
        # STAGE 2: FILTERING & BUSINESS LOGIC
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        filtered_items = self.filtering_stage.filter(
            candidates=candidates,
            consumed_item_ids=set(consumed_items or []),
            max_candidates=100,
        )
        t1 = time.perf_counter()
        stage_metrics.append(
            StageMetric(
                stage_name="Business Filtering",
                stage_number=2,
                input_count=len(candidates),
                output_count=len(filtered_items),
                latency_ms=round((t1 - t0) * 1000.0, 2),
                description="Deduplication against history, age rating, and content safety filters",
            )
        )

        # -------------------------------------------------------------
        # STAGE 3: HEAVY RANKING & MULTI-TASK SCORING
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        ranked_items = self.ranking_stage.rank(
            user_id=user_id,
            candidates=filtered_items,
            budget=50,
            context=ctx,
        )
        t1 = time.perf_counter()
        stage_metrics.append(
            StageMetric(
                stage_name="Heavy Ranking",
                stage_number=3,
                input_count=len(filtered_items),
                output_count=len(ranked_items),
                latency_ms=round((t1 - t0) * 1000.0, 2),
                description="MMoE multi-task deep scoring balancing CTR, Watch Time, and Rating",
            )
        )

        # -------------------------------------------------------------
        # STAGE 4: RE-RANKING & MMR DIVERSITY
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        reranked_items = self.reranking_stage.rerank(
            ranked_items=ranked_items,
            item_embeddings=item_embeddings,
            top_k=20,
        )
        t1 = time.perf_counter()
        stage_metrics.append(
            StageMetric(
                stage_name="Re-Ranking & Diversity",
                stage_number=4,
                input_count=len(ranked_items),
                output_count=len(reranked_items),
                latency_ms=round((t1 - t0) * 1000.0, 2),
                description="Maximal Marginal Relevance (MMR) Intra-List Diversity & Exploration",
            )
        )

        # -------------------------------------------------------------
        # STAGE 5: SLATE GENERATION & ARTWORK PERSONALIZATION
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if user_genre_vector is None:
            user_genre_vector = np.array([0.4, 0.3, 0.1, 0.2], dtype=np.float32)

        slate_page = self.slate_stage.generate_slate_page(
            user_id=user_id,
            domain=domain,
            items=reranked_items,
            user_context_vec=user_genre_vector,
        )
        t1 = time.perf_counter()
        stage_metrics.append(
            StageMetric(
                stage_name="Slate & Artwork",
                stage_number=5,
                input_count=len(reranked_items),
                output_count=len(reranked_items),
                latency_ms=round((t1 - t0) * 1000.0, 2),
                description="Multi-row dynamic page assembly with LinUCB personalized thumbnail selection",
            )
        )

        total_latency = (time.perf_counter() - start_pipeline) * 1000.0

        return PipelineResult(
            user_id=user_id,
            domain=domain,
            total_latency_ms=round(total_latency, 2),
            stage_metrics=stage_metrics,
            slate_page=slate_page,
            final_items=reranked_items,
        )
