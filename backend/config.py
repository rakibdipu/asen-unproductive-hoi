"""
Omni-RecSys Platform - Centralized Configuration & Hyperparameters
Supports all 6 generations of recommendation algorithms and multi-domain datasets.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Dict, List, Any


class Settings(BaseSettings):
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "saved_models"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000", "*"]

    # LLM Settings (for Generative & Cold-Start)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Default RecSys Hyperparameters
    EMBEDDING_DIM: int = 64
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 10
    TOP_K: int = 10

    # Pipeline Thresholds
    STAGE1_RETRIEVAL_BUDGET: int = 200
    STAGE2_FILTERED_BUDGET: int = 100
    STAGE3_RANKING_BUDGET: int = 50
    STAGE4_RERANK_BUDGET: int = 20
    STAGE5_SLATE_ROWS: int = 5
    STAGE5_ITEMS_PER_ROW: int = 10

    # Diversity & Bandit Settings
    MMR_LAMBDA: float = 0.65  # Balance between relevance (1.0) and diversity (0.0)
    LINUCB_ALPHA: float = 0.25  # Exploration bonus parameter
    THOMPSON_PRIOR_ALPHA: float = 1.0
    THOMPSON_PRIOR_BETA: float = 1.0

    # Domain Configurations
    SUPPORTED_DOMAINS: List[str] = ["movies", "music", "ecommerce", "short_feed"]

    # Model Zoo Registry
    REGISTERED_MODELS: Dict[str, Dict[str, Any]] = {
        "popularity": {
            "name": "Time-Decayed Popularity",
            "generation": "Gen 1 (Heuristic)",
            "company": "Baseline / Classic",
            "year": 1990,
            "description": "Exponential time-decayed interaction popularity scoring."
        },
        "item_cf": {
            "name": "Item-Item Collaborative Filtering",
            "generation": "Gen 1 (Heuristic)",
            "company": "Amazon (Sarwar et al.)",
            "year": 2001,
            "description": "Cosine similarity over user interaction vectors with k-nearest neighbors."
        },
        "implicit_als": {
            "name": "Implicit ALS (Alternating Least Squares)",
            "generation": "Gen 2 (Latent Factor)",
            "company": "AT&T / Yahoo (Hu, Koren, Volinsky)",
            "year": 2008,
            "description": "Factorizes implicit feedback matrix with confidence weighting."
        },
        "bpr_mf": {
            "name": "BPR-MF (Bayesian Personalized Ranking)",
            "generation": "Gen 2 (Latent Factor)",
            "company": "U. of Hildesheim (Rendle et al.)",
            "year": 2009,
            "description": "Pairwise ranking optimization maximizing margin between positive and negative items."
        },
        "two_tower": {
            "name": "Two-Tower Deep Neural Network",
            "generation": "Gen 3 (Deep Retrieval)",
            "company": "Google / YouTube (Covington et al.)",
            "year": 2016,
            "description": "Dual user and item neural towers mapped to common metric space with ANN."
        },
        "wide_deep": {
            "name": "Wide & Deep Learning",
            "generation": "Gen 4 (Deep Ranking)",
            "company": "Google Play (Cheng et al.)",
            "year": 2016,
            "description": "Jointly trains wide linear models (memorization) and deep neural nets (generalization)."
        },
        "deepfm": {
            "name": "DeepFM (Deep Factorization Machine)",
            "generation": "Gen 4 (Deep Ranking)",
            "company": "Huawei / CAS (Guo et al.)",
            "year": 2017,
            "description": "Integrates factorization machines with deep neural networks without manual feature engineering."
        },
        "din": {
            "name": "DIN (Deep Interest Network)",
            "generation": "Gen 4 (Deep Ranking)",
            "company": "Alibaba (Zhou et al.)",
            "year": 2018,
            "description": "Local activation attention unit to capture dynamic user interests relative to candidate item."
        },
        "mmoe": {
            "name": "MMoE (Multi-gate Mixture-of-Experts)",
            "generation": "Gen 4 (Multi-Task Ranking)",
            "company": "Google (Ma et al.)",
            "year": 2018,
            "description": "Shared expert sub-networks with task-specific gates for CTR, Watch-Time, and Rating."
        },
        "sasrec": {
            "name": "SASRec (Self-Attentive Sequential Recommendation)",
            "generation": "Gen 5 (Sequential Transformer)",
            "company": "UCSD (Kang & McAuley)",
            "year": 2018,
            "description": "Self-attention mechanism to model user sequential behavior and session drift."
        },
        "lightgcn": {
            "name": "LightGCN (Simplified Graph Convolution)",
            "generation": "Gen 5 (Graph Neural Network)",
            "company": "NUS (He et al.)",
            "year": 2020,
            "description": "Linear neighborhood aggregation over bipartite user-item graph without non-linearities."
        },
        "linucb": {
            "name": "LinUCB Contextual Bandit (Artwork Personalization)",
            "generation": "Gen 6 (Bandits & RL)",
            "company": "Netflix (Amat Gil et al.)",
            "year": 2018,
            "description": "Ridge regression with upper confidence bound for dynamic thumbnail exploration/exploitation."
        },
        "llm_reranker": {
            "name": "LLM Generative Recommender & Explainer",
            "generation": "Gen 6 (Generative & Foundation)",
            "company": "Meta HSTU / Gemini / Netflix 2026",
            "year": 2026,
            "description": "Semantic cold-start matching, in-context reranking, and natural language explanations."
        }
    }

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
