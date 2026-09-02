"""
Omni-RecSys Model Registry & Factory
Provides seamless instantiation and training of all 12 algorithms across 6 generations.
"""

from typing import Dict, Type, Optional, Any
from backend.models.base import BaseRecommender
from backend.models.popularity import PopularityRecommender
from backend.models.item_cf import ItemCFRecommender
from backend.models.implicit_als import ImplicitALSRecommender
from backend.models.bpr_mf import BPRMFRecommender
from backend.models.two_tower import TwoTowerRecommender
from backend.models.wide_deep import WideAndDeepRecommender
from backend.models.deepfm import DeepFMRecommender
from backend.models.din import DINRecommender
from backend.models.mmoe import MMoERecommender
from backend.models.sasrec import SASRecRecommender
from backend.models.lightgcn import LightGCNRecommender
from backend.models.llm_reranker import LLMRecommender

MODEL_REGISTRY: Dict[str, Type[BaseRecommender]] = {
    "popularity": PopularityRecommender,
    "item_cf": ItemCFRecommender,
    "implicit_als": ImplicitALSRecommender,
    "bpr_mf": BPRMFRecommender,
    "two_tower": TwoTowerRecommender,
    "wide_deep": WideAndDeepRecommender,
    "deepfm": DeepFMRecommender,
    "din": DINRecommender,
    "mmoe": MMoERecommender,
    "sasrec": SASRecRecommender,
    "lightgcn": LightGCNRecommender,
    "llm_reranker": LLMRecommender,
}


def get_model(model_name: str, domain: str = "movies", config: Optional[Dict[str, Any]] = None) -> BaseRecommender:
    """Instantiate a recommendation model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(domain=domain, config=config)
