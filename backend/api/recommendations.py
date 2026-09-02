"""
Recommendation API Endpoints
Provides single-algorithm testing and full 5-stage production pipeline execution.
"""

import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends

from backend.models import get_model, MODEL_REGISTRY
from backend.models.base import RecommendationResult, ItemScore
from backend.data.loaders import dataset_manager
from backend.pipeline.orchestrator import ProductionPipeline, PipelineResult
from backend.bandits.linucb import LinUCBBandit

router = APIRouter(prefix="/api/recommend", tags=["Recommendations"])

# Global in-memory cache of initialized models per domain
DOMAINS_CACHE: Dict[str, Dict[str, Any]] = {}
PIPELINES_CACHE: Dict[str, ProductionPipeline] = {}
SHARED_ARTWORK_BANDIT = LinUCBBandit(
    arm_names=["variant_action", "variant_character", "variant_romantic", "variant_cinematic"],
    feature_dim=4,
    alpha=0.3,
)


def get_domain_models(domain: str = "movies") -> Dict[str, Any]:
    if domain not in DOMAINS_CACHE:
        dataset = dataset_manager.load_domain(domain)
        models = {}
        # Pre-fit key models for the domain
        for key in ["popularity", "item_cf", "implicit_als", "two_tower", "deepfm", "din", "mmoe", "sasrec", "lightgcn", "llm_reranker"]:
            m = get_model(key, domain=domain)
            m.fit(dataset.interactions_df, dataset.items_df)
            models[key] = m
        DOMAINS_CACHE[domain] = models
    return DOMAINS_CACHE[domain]


def get_pipeline(domain: str = "movies") -> ProductionPipeline:
    if domain not in PIPELINES_CACHE:
        models = get_domain_models(domain)
        retrieval_models = {
            "two_tower": models["two_tower"],
            "implicit_als": models["implicit_als"],
            "sasrec": models["sasrec"],
            "lightgcn": models["lightgcn"],
            "popularity": models["popularity"],
        }
        ranking_model = models.get("mmoe", models["deepfm"])
        pipeline = ProductionPipeline(
            retrieval_models=retrieval_models,
            ranking_model=ranking_model,
            artwork_bandit=SHARED_ARTWORK_BANDIT,
        )
        PIPELINES_CACHE[domain] = pipeline
    return PIPELINES_CACHE[domain]


class SingleModelRequest(BaseModel):
    user_id: str
    algorithm: str = "sasrec"
    domain: str = "movies"
    n: int = 10
    exclude_consumed: bool = True
    context: Dict[str, Any] = Field(default_factory=dict)


class PipelineRequest(BaseModel):
    user_id: str
    domain: str = "movies"
    context: Dict[str, Any] = Field(default_factory=dict)


@router.post("/single", response_model=RecommendationResult)
def recommend_single(req: SingleModelRequest):
    t0 = time.perf_counter()
    models = get_domain_models(req.domain)
    if req.algorithm not in models:
        # Lazy fit if not pre-fitted
        dataset = dataset_manager.load_domain(req.domain)
        m = get_model(req.algorithm, domain=req.domain)
        m.fit(dataset.interactions_df, dataset.items_df)
        models[req.algorithm] = m

    model = models[req.algorithm]
    dataset = dataset_manager.load_domain(req.domain)

    exclude = []
    if req.exclude_consumed:
        history = dataset.get_user_history(req.user_id)
        exclude = history["item_id"].tolist() if not history.empty else []

    items = model.recommend(user_id=req.user_id, n=req.n, exclude_item_ids=exclude, context=req.context)
    latency = (time.perf_counter() - t0) * 1000.0

    return RecommendationResult(
        user_id=req.user_id,
        domain=req.domain,
        algorithm=req.algorithm,
        total_candidates_evaluated=len(dataset.items_df),
        latency_ms=round(latency, 2),
        items=items,
    )


@router.post("/pipeline", response_model=PipelineResult)
def recommend_pipeline(req: PipelineRequest):
    pipeline = get_pipeline(req.domain)
    dataset = dataset_manager.load_domain(req.domain)

    history = dataset.get_user_history(req.user_id)
    consumed = history["item_id"].tolist() if not history.empty else []

    genres = ["Action", "Sci-Fi", "Drama", "Comedy"]
    from backend.streaming.feature_store import feature_store
    user_vec = feature_store.get_user_genre_vector(req.user_id, genres)

    res = pipeline.execute(
        user_id=req.user_id,
        domain=req.domain,
        catalog_size=len(dataset.items_df),
        consumed_items=consumed,
        context=req.context,
        user_genre_vector=user_vec,
    )
    return res
