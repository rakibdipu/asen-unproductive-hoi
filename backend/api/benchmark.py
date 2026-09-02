"""
Benchmark & Model Arena API Endpoints
Runs head-to-head algorithm tournaments across NDCG, Coverage, Diversity, and Latency.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter
from backend.evaluation.arena import ModelArena, ArenaBenchmarkResponse
from backend.data.loaders import dataset_manager
from backend.api.recommendations import get_domain_models

router = APIRouter(prefix="/api/benchmark", tags=["Benchmark & Arena"])


class TournamentRequest(BaseModel):
    domain: str = "movies"
    models: Optional[List[str]] = Field(
        default=["popularity", "item_cf", "implicit_als", "two_tower", "deepfm", "din", "mmoe", "sasrec", "lightgcn"]
    )
    k: int = 10
    sample_users: int = 25


@router.post("/arena", response_model=ArenaBenchmarkResponse)
def run_tournament(req: TournamentRequest):
    dataset = dataset_manager.load_domain(req.domain)
    models = get_domain_models(req.domain)
    arena = ModelArena(dataset=dataset, trained_models=models)
    return arena.run_tournament(model_keys=req.models, k=req.k, sample_users=req.sample_users)
