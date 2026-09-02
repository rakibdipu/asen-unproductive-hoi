"""
Explainability & Attention Explorer Endpoints
Extracts target attention weights from DIN, feature importance from DeepFM,
and natural language rationales from LLM.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from backend.models.base import ExplainResult
from backend.api.recommendations import get_domain_models

router = APIRouter(prefix="/api/explain", tags=["Explainability"])


class ExplainRequest(BaseModel):
    user_id: str
    item_id: str
    algorithm: str = "din"
    domain: str = "movies"


@router.post("", response_model=ExplainResult)
def explain_recommendation(req: ExplainRequest):
    models = get_domain_models(req.domain)
    if req.algorithm not in models:
        raise HTTPException(status_code=400, detail=f"Algorithm '{req.algorithm}' not found or not trained.")

    model = models[req.algorithm]
    return model.explain(user_id=req.user_id, item_id=req.item_id)
