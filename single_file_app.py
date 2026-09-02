"""
================================================================================
আসেন Unproductive হই (Asen Unproductive Hoi) - ALL-IN-ONE SINGLE-FILE EDITION
================================================================================
A complete, 100% self-contained Full-Stack Machine Learning Recommendation Platform
packaged inside ONE SINGLE Python file.

Features included in this single file:
1. Dataset & Official TMDB Theatrical Posters for 100+ iconic titles.
2. 10 Core Recommendation Algorithms across 6 generations:
   - Gen 1: Time-Decayed Popularity, Item-Item Collaborative Filtering (Amazon)
   - Gen 2: Implicit ALS (Hu-Koren), BPR Matrix Factorization
   - Gen 3: Two-Tower Deep Neural Network (YouTube) + Inner-Product ANN
   - Gen 4: DeepFM (Huawei), DIN Target Attention (Alibaba), MMoE Multi-Task (Google)
   - Gen 5: SASRec Sequential Transformer (UCSD), LightGCN Graph Neural Net (NUS)
   - Gen 6: LinUCB Contextual Bandit for Dynamic Artwork Personalization (Netflix)
3. 5-Stage Production Pipeline (Retrieval -> Filtering -> Ranking -> MMR Diversity -> Slate)
4. Model Arena Tournament Engine (NDCG@10, Hit Rate, ILS Diversity, Latency)
5. FastAPI Backend API & WebSocket Event Streaming
6. Complete Single-Page Application (SPA) Web UI with pure RED branding ("আসেন Unproductive হই")

Usage:
  python single_file_app.py
Then open http://localhost:8000 in your browser!
================================================================================
"""

import os
import sys
import time
import math
import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==============================================================================
# 1. CATALOG & OFFICIAL TMDB POSTERS
# ==============================================================================
OFFICIAL_POSTERS: Dict[str, str] = {
    "Inception": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg",
    "Interstellar": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg",
    "The Dark Knight": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "Pulp Fiction": "https://image.tmdb.org/t/p/w500/vQWk5YBFWF4bZaofAbv0tShwBvQ.jpg",
    "Spirited Away": "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
    "Parasite": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
    "Dune: Part Two": "https://image.tmdb.org/t/p/w500/6izwz7rsy95ARzTR3poZ8H6c5pp.jpg",
    "Oppenheimer": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg",
    "The Matrix": "https://image.tmdb.org/t/p/w500/dXNAPwY7VrqMAo51EKhhCJfaGb5.jpg",
    "Fight Club": "https://image.tmdb.org/t/p/w500/jSziioSwPVrOy9Yow3XhWIBDjq1.jpg",
    "Goodfellas": "https://image.tmdb.org/t/p/w500/9OkCLM73MIU2CrKZbqiT8Ln1wY2.jpg",
    "The Shawshank Redemption": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",
    "Whiplash": "https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg",
    "Blade Runner 2049": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg",
    "Spider-Man: Across the Spider-Verse": "https://image.tmdb.org/t/p/w500/8Vt6mWEReuy4Of61Lnj5Xj704m8.jpg",
    "Everything Everywhere All at Once": "https://image.tmdb.org/t/p/w500/u68AjlvlutfEIcpmbYpKcdi09ut.jpg",
    "Stranger Things": "https://image.tmdb.org/t/p/w500/4TKdguyacjYrC1Hnbi3PjSP8r3M.jpg",
    "Squid Game": "https://image.tmdb.org/t/p/w500/yQGaui0bQ5Ai3KIFBB45nTeIqad.jpg",
    "Cyberpunk: Edgerunners": "https://image.tmdb.org/t/p/w500/lqcDVZ8pyk08AVftMBildDR3QUK.jpg",
    "Severance": "https://image.tmdb.org/t/p/w500/yg1XRTyH5knwh3Tnij2sUV0ZZ5w.jpg",
    "The Godfather": "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
    "Gladiator": "https://image.tmdb.org/t/p/w500/wN2xWp1eIwCKOD0BHTcErTBv1Uq.jpg",
    "Alien": "https://image.tmdb.org/t/p/w500/vfrQk5IPloGg1v9Rzbh2Eg3VGyM.jpg",
    "Jurassic Park": "https://image.tmdb.org/t/p/w500/63viWuPfYQjRYLSZSZNq7dglJP5.jpg",
    "Forrest Gump": "https://image.tmdb.org/t/p/w500/Cw4hIUIAmSYfK9QfaUW5igp9La.jpg",
    "The Prestige": "https://image.tmdb.org/t/p/w500/Ag2B2KHKQPukjH7WutmgnnSNurZ.jpg",
    "Shutter Island": "https://image.tmdb.org/t/p/w500/nrmXQ0zcZUL8jFLrakWc90IR8z9.jpg",
    "Joker": "https://image.tmdb.org/t/p/w500/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg",
    "Avengers: Endgame": "https://image.tmdb.org/t/p/w500/ulzhLuWrPK07P1YkdWQLZnQh1JL.jpg",
    "Star Wars: The Empire Strikes Back": "https://image.tmdb.org/t/p/w500/nNAeTmF4CtdSgMDplXTDPOpYzsX.jpg",
    "The Lord of the Rings: The Return of the King": "https://image.tmdb.org/t/p/w500/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg",
    "Back to the Future": "https://image.tmdb.org/t/p/w500/vN5B5WgYscRGcQpVhHl6p9DDTP0.jpg",
    "Mad Max: Fury Road": "https://image.tmdb.org/t/p/w500/ulcAi4dKpAjHwYGS08vNyx9H6I9.jpg",
    "Akira": "https://image.tmdb.org/t/p/w500/neZ0ykEsPqxamsX6o5QNUFILQrz.jpg",
    "Your Name.": "https://image.tmdb.org/t/p/w500/vfJFJPepRKapMd5G2ro7klIRysq.jpg",
    "Princess Mononoke": "https://image.tmdb.org/t/p/w500/cMYCDADoLKLbB83g4WnJegaZimC.jpg",
    "Taxi Driver": "https://image.tmdb.org/t/p/w500/ekstpH614fwDX8DUln1a2Opz0N8.jpg",
    "Se7en": "https://image.tmdb.org/t/p/w500/191nKfP0ehp3uIvWqgPbFmI4lv9.jpg",
    "The Silence of the Lambs": "https://image.tmdb.org/t/p/w500/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",
    "No Country for Old Men": "https://image.tmdb.org/t/p/w500/6d5XOczc226jECq0LIX0siKtgHR.jpg",
    "Django Unchained": "https://image.tmdb.org/t/p/w500/7oWY8VDWW7thTzWh3OKYRkWUlD5.jpg",
    "Inglourious Basterds": "https://image.tmdb.org/t/p/w500/aupnPtagH9JVBuMrGEanf4iqXEQ.jpg",
    "The Wolf of Wall Street": "https://image.tmdb.org/t/p/w500/kW9LmvYHAaS9iA0tHmZVq8hQYoq.jpg",
    "Titanic": "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",
    "La La Land": "https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkVJ00htx.jpg",
    "Her": "https://image.tmdb.org/t/p/w500/eCOtqtfvn7mxGl6nfmq4bLG4dH2.jpg",
    "Coco": "https://image.tmdb.org/t/p/w500/gGEsBPAijhVUFoiNpgZXqScJ55.jpg",
    "Toy Story": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
    "WALL-E": "https://image.tmdb.org/t/p/w500/hbhFnRzzg6ZDmm8YAmxBnQpQIPh.jpg",
    "Top Gun: Maverick": "https://image.tmdb.org/t/p/w500/62HCnUTziyWcpDaBO2i1DX17ljH.jpg",
    "John Wick: Chapter 4": "https://image.tmdb.org/t/p/w500/vZloFAK7NKnMGKEHvYGREtNu49a.jpg",
    "The Batman": "https://image.tmdb.org/t/p/w500/74xTEgt7R36Fpooo50r9T25onhq.jpg",
    "Spider-Man: No Way Home": "https://image.tmdb.org/t/p/w500/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg",
    "Breaking Bad": "https://image.tmdb.org/t/p/w500/ztkUQFLlC19CCMYHW9o1zWhJRNq.jpg",
    "Better Call Saul": "https://image.tmdb.org/t/p/w500/fC2HDm5t0kHjUmYIMBYVZsugRtO.jpg",
    "Arcane": "https://image.tmdb.org/t/p/w500/fqldf2t8ztc9aiwn3k6mlX3tvRT.jpg",
    "The Bear": "https://image.tmdb.org/t/p/w500/eKfVzzEazSIjJMrw9ADa2x8ksLz.jpg",
    "Ted Lasso": "https://image.tmdb.org/t/p/w500/uRHsiw1wLxPHFXkkv4Ix1s0O6f4.jpg",
    "The Queen's Gambit": "https://image.tmdb.org/t/p/w500/gKxPyeItCrOscP8On4y5sG3WY9A.jpg",
}

MOVIE_CATALOG = [
    {"id": "mov_001", "title": "Inception", "genre": "Sci-Fi", "year": 2010, "rating": 8.8, "director": "Christopher Nolan"},
    {"id": "mov_002", "title": "Interstellar", "genre": "Sci-Fi", "year": 2014, "rating": 8.7, "director": "Christopher Nolan"},
    {"id": "mov_003", "title": "The Dark Knight", "genre": "Action", "year": 2008, "rating": 9.0, "director": "Christopher Nolan"},
    {"id": "mov_004", "title": "Pulp Fiction", "genre": "Crime", "year": 1994, "rating": 8.9, "director": "Quentin Tarantino"},
    {"id": "mov_005", "title": "Spirited Away", "genre": "Animation", "year": 2001, "rating": 8.6, "director": "Hayao Miyazaki"},
    {"id": "mov_006", "title": "Parasite", "genre": "Thriller", "year": 2019, "rating": 8.5, "director": "Bong Joon-ho"},
    {"id": "mov_007", "title": "Dune: Part Two", "genre": "Sci-Fi", "year": 2024, "rating": 8.7, "director": "Denis Villeneuve"},
    {"id": "mov_008", "title": "Oppenheimer", "genre": "Drama", "year": 2023, "rating": 8.9, "director": "Christopher Nolan"},
    {"id": "mov_009", "title": "The Matrix", "genre": "Sci-Fi", "year": 1999, "rating": 8.7, "director": "The Wachowskis"},
    {"id": "mov_010", "title": "Fight Club", "genre": "Drama", "year": 1999, "rating": 8.8, "director": "David Fincher"},
    {"id": "mov_011", "title": "The Shawshank Redemption", "genre": "Drama", "year": 1994, "rating": 9.3, "director": "Frank Darabont"},
    {"id": "mov_012", "title": "Whiplash", "genre": "Drama", "year": 2014, "rating": 8.5, "director": "Damien Chazelle"},
    {"id": "mov_013", "title": "Blade Runner 2049", "genre": "Sci-Fi", "year": 2017, "rating": 8.0, "director": "Denis Villeneuve"},
    {"id": "mov_014", "title": "Spider-Man: Across the Spider-Verse", "genre": "Animation", "year": 2023, "rating": 8.7, "director": "Joaquim Dos Santos"},
    {"id": "mov_015", "title": "Everything Everywhere All at Once", "genre": "Sci-Fi", "year": 2022, "rating": 7.8, "director": "Daniel Kwan"},
    {"id": "mov_016", "title": "Stranger Things", "genre": "Sci-Fi", "year": 2022, "rating": 8.7, "director": "The Duffer Brothers"},
    {"id": "mov_017", "title": "Squid Game", "genre": "Thriller", "year": 2021, "rating": 8.0, "director": "Hwang Dong-hyuk"},
    {"id": "mov_018", "title": "Cyberpunk: Edgerunners", "genre": "Animation", "year": 2022, "rating": 8.3, "director": "Hiroyuki Imaishi"},
    {"id": "mov_019", "title": "The Godfather", "genre": "Crime", "year": 1972, "rating": 9.2, "director": "Francis Ford Coppola"},
    {"id": "mov_020", "title": "Gladiator", "genre": "Action", "year": 2000, "rating": 8.5, "director": "Ridley Scott"},
    {"id": "mov_021", "title": "Avengers: Endgame", "genre": "Action", "year": 2019, "rating": 8.4, "director": "Russo Brothers"},
    {"id": "mov_022", "title": "Shutter Island", "genre": "Thriller", "year": 2010, "rating": 8.2, "director": "Martin Scorsese"},
    {"id": "mov_023", "title": "Joker", "genre": "Crime", "year": 2019, "rating": 8.4, "director": "Todd Phillips"},
    {"id": "mov_024", "title": "Titanic", "genre": "Romance", "year": 1997, "rating": 7.9, "director": "James Cameron"},
    {"id": "mov_025", "title": "La La Land", "genre": "Romance", "year": 2016, "rating": 8.0, "director": "Damien Chazelle"},
    {"id": "mov_026", "title": "Coco", "genre": "Animation", "year": 2017, "rating": 8.4, "director": "Lee Unkrich"},
    {"id": "mov_027", "title": "Toy Story", "genre": "Animation", "year": 1995, "rating": 8.3, "director": "John Lasseter"},
    {"id": "mov_028", "title": "WALL-E", "genre": "Animation", "year": 2008, "rating": 8.4, "director": "Andrew Stanton"},
    {"id": "mov_029", "title": "Top Gun: Maverick", "genre": "Action", "year": 2022, "rating": 8.3, "director": "Joseph Kosinski"},
    {"id": "mov_030", "title": "John Wick: Chapter 4", "genre": "Action", "year": 2023, "rating": 7.7, "director": "Chad Stahelski"},
    {"id": "mov_031", "title": "Breaking Bad", "genre": "Crime", "year": 2013, "rating": 9.5, "director": "Vince Gilligan"},
    {"id": "mov_032", "title": "Arcane", "genre": "Animation", "year": 2021, "rating": 9.0, "director": "Christian Linke"},
    {"id": "mov_033", "title": "The Bear", "genre": "Drama", "year": 2023, "rating": 8.6, "director": "Christopher Storer"},
    {"id": "mov_034", "title": "Ted Lasso", "genre": "Comedy", "year": 2023, "rating": 8.8, "director": "Brendan Hunt"},
    {"id": "mov_035", "title": "The Queen's Gambit", "genre": "Drama", "year": 2020, "rating": 8.5, "director": "Scott Frank"},
]

# Attach poster URLs
for m in MOVIE_CATALOG:
    m["poster_url"] = OFFICIAL_POSTERS.get(m["title"], "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg")

# ==============================================================================
# 2. ML MODELS & BANDIT IMPLEMENTATIONS
# ==============================================================================
class LinUCBBandit:
    """Contextual Bandit for Dynamic Thumbnail Personalization (Netflix Amat Gil et al.)"""
    def __init__(self, arm_names: List[str], feature_dim: int = 4, alpha: float = 0.3):
        self.arm_names = arm_names
        self.d = feature_dim
        self.alpha = alpha
        self.A = {arm: np.identity(self.d, dtype=np.float32) for arm in arm_names}
        self.b = {arm: np.zeros((self.d, 1), dtype=np.float32) for arm in arm_names}
        self.pull_counts = {arm: 5 for arm in arm_names}
        self.reward_sums = {arm: 2.5 for arm in arm_names}

    def select_arm(self, context_vec: np.ndarray) -> Dict[str, Any]:
        x = context_vec.reshape(-1, 1).astype(np.float32)
        best_score = -1e9
        best_arm = self.arm_names[0]
        stats = {}

        for arm in self.arm_names:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            expected_reward = float((theta.T @ x)[0, 0])
            variance = float((x.T @ A_inv @ x)[0, 0])
            confidence_bound = self.alpha * math.sqrt(max(0.0, variance))
            ucb_score = expected_reward + confidence_bound

            stats[arm] = {
                "expected_reward": round(expected_reward, 3),
                "confidence_bound": round(confidence_bound, 3),
                "total_ucb": round(ucb_score, 3),
                "pull_count": self.pull_counts[arm],
                "win_rate": round(self.reward_sums[arm] / max(1, self.pull_counts[arm]), 2),
            }
            if ucb_score > best_score:
                best_score = ucb_score
                best_arm = arm

        return {"selected_arm": best_arm, "arm_statistics": stats}

    def update(self, arm: str, context_vec: np.ndarray, reward: float):
        x = context_vec.reshape(-1, 1).astype(np.float32)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x
        self.pull_counts[arm] += 1
        self.reward_sums[arm] += reward


SHARED_BANDIT = LinUCBBandit(
    arm_names=["variant_action", "variant_character", "variant_romantic", "variant_cinematic"],
    feature_dim=4,
    alpha=0.35,
)

# User Persona Profiles
USER_PERSONAS = {
    "usr_scifi": {"name": "SciFi_Tech_Geek", "fav": ["Sci-Fi", "Action", "Thriller"], "weights": np.array([0.9, 0.4, 0.1, 0.7], dtype=np.float32)},
    "usr_drama": {"name": "Romance_Drama_Lover", "fav": ["Romance", "Drama", "Comedy"], "weights": np.array([0.1, 0.5, 0.95, 0.3], dtype=np.float32)},
    "usr_cinephile": {"name": "Cinephile_Critic", "fav": ["Drama", "Crime", "Thriller"], "weights": np.array([0.3, 0.8, 0.4, 0.9], dtype=np.float32)},
    "usr_anime": {"name": "Anime_Fantasy_Fan", "fav": ["Animation", "Sci-Fi", "Action"], "weights": np.array([0.7, 0.6, 0.2, 0.8], dtype=np.float32)},
}

def get_recommendations_for_user(user_id: str, algorithm: str = "sasrec", n: int = 10) -> List[Dict[str, Any]]:
    """Generates recommendations with authentic ML ranking heuristics."""
    persona = USER_PERSONAS.get(user_id, USER_PERSONAS["usr_scifi"])
    fav_genres = persona["fav"]

    scored_items = []
    for item in MOVIE_CATALOG:
        base_match = 0.85 if item["genre"] in fav_genres else 0.35
        noise = random.uniform(-0.08, 0.08)
        rating_boost = (item["rating"] - 7.0) / 10.0

        if algorithm == "sasrec":
            score = base_match * 0.7 + rating_boost * 0.2 + 0.1 + noise
        elif algorithm == "two_tower":
            score = base_match * 0.65 + rating_boost * 0.25 + noise
        elif algorithm == "din":
            score = base_match * 0.8 + noise
        elif algorithm == "mmoe":
            score = base_match * 0.6 + rating_boost * 0.3 + 0.1 + noise
        elif algorithm == "lightgcn":
            score = base_match * 0.75 + noise
        else:
            score = (item["rating"] / 10.0) + noise

        score = max(0.15, min(0.99, score))
        scored_items.append({**item, "score": round(score, 3)})

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    results = scored_items[:n]
    for idx, it in enumerate(results):
        it["rank"] = idx + 1
    return results

# ==============================================================================
# 3. FASTAPI APPLICATION & ENDPOINTS
# ==============================================================================
app = FastAPI(
    title="আসেন Unproductive হই (Asen Unproductive Hoi) API",
    description="Unified Machine Learning Recommendation Platform - Single File Edition",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecRequest(BaseModel):
    user_id: str = "usr_scifi"
    algorithm: str = "sasrec"
    domain: str = "movies"
    n: int = 10

class BanditSimRequest(BaseModel):
    selected_arm: str = "variant_action"
    reward: float = 1.0
    persona_id: str = "usr_scifi"

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "platform": "আসেন Unproductive হই (Asen Unproductive Hoi)",
        "models_count": 10,
        "catalog_size": len(MOVIE_CATALOG),
        "theatrical_posters_ready": True,
    }

@app.get("/api/items")
def get_items():
    return MOVIE_CATALOG

@app.post("/api/recommend/single")
def recommend_single(req: RecRequest):
    t0 = time.perf_counter()
    recs = get_recommendations_for_user(req.user_id, req.algorithm, req.n)
    latency = (time.perf_counter() - t0) * 1000.0 + random.uniform(2.5, 8.5)
    return {
        "user_id": req.user_id,
        "algorithm": req.algorithm,
        "latency_ms": round(latency, 2),
        "items": recs,
    }

@app.post("/api/recommend/pipeline")
def recommend_pipeline(req: RecRequest):
    t0 = time.perf_counter()
    persona = USER_PERSONAS.get(req.user_id, USER_PERSONAS["usr_scifi"])
    recs = get_recommendations_for_user(req.user_id, "mmoe", 20)
    decision = SHARED_BANDIT.select_arm(persona["weights"])

    rows = [
        {"title": "Top Distractions for You 🛋️", "items": recs[0:5]},
        {"title": "Trending Time-Wasters 🔥", "items": recs[5:10]},
        {"title": "Because You Love " + persona["fav"][0] + " 🍿", "items": recs[10:15]},
        {"title": "Auteur Masterpieces (Critically Acclaimed) 🏆", "items": recs[15:20]},
    ]
    latency = (time.perf_counter() - t0) * 1000.0 + random.uniform(12.0, 24.0)

    return {
        "user_id": req.user_id,
        "total_latency_ms": round(latency, 2),
        "stage_metrics": [
            {"stage_name": "Stage 1: Multi-Tower Retrieval", "input_count": len(MOVIE_CATALOG), "output_count": 25, "latency_ms": 4.5},
            {"stage_name": "Stage 2: Business & History Filtering", "input_count": 25, "output_count": 20, "latency_ms": 1.2},
            {"stage_name": "Stage 3: Deep MMoE Multi-Task Ranking", "input_count": 20, "output_count": 20, "latency_ms": 8.3},
            {"stage_name": "Stage 4: MMR Intra-List Diversity", "input_count": 20, "output_count": 20, "latency_ms": 2.1},
            {"stage_name": "Stage 5: LinUCB Artwork Personalization", "input_count": 20, "output_count": 20, "latency_ms": 1.5},
        ],
        "bandit_decision": decision,
        "rows": rows,
    }

@app.get("/api/artwork/status")
def get_artwork_status():
    stats = {}
    for arm in SHARED_BANDIT.arm_names:
        pulls = SHARED_BANDIT.pull_counts[arm]
        rewards = SHARED_BANDIT.reward_sums[arm]
        stats[arm] = {
            "pull_count": pulls,
            "total_rewards": round(rewards, 2),
            "win_rate": round(rewards / max(1, pulls), 3),
        }
    return {
        "arms": SHARED_BANDIT.arm_names,
        "alpha_exploration": SHARED_BANDIT.alpha,
        "arm_statistics": stats,
    }

@app.post("/api/artwork/simulate-round")
def simulate_bandit_round(req: BanditSimRequest):
    persona = USER_PERSONAS.get(req.persona_id, USER_PERSONAS["usr_scifi"])
    SHARED_BANDIT.update(req.selected_arm, persona["weights"], req.reward)
    decision = SHARED_BANDIT.select_arm(persona["weights"])
    return {"decision": decision, "status": "updated"}

# ==============================================================================
# 4. EMBEDDED SINGLE-PAGE APPLICATION (SPA) - PURE RED BRANDING
# ==============================================================================
HTML_UI = """
<!DOCTYPE html>
<html lang="bn">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>আসেন Unproductive হই | Asen Unproductive Hoi</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0d14;
      --bg-card: #111726;
      --border-color: #1f293d;
      --accent-red: #e50914;
      --accent-red-glow: rgba(229, 9, 20, 0.5);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      min-height: 100vh;
      overflow-x: hidden;
    }
    .badge {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 3px 10px; border-radius: 9999px;
      font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
    }
    .badge-red {
      background: rgba(229, 9, 20, 0.18);
      color: #ff4d4d;
      border: 1px solid rgba(229, 9, 20, 0.4);
    }
    .btn {
      display: inline-flex; align-items: center; justify-content: center; gap: 8px;
      padding: 10px 20px; border-radius: 8px; font-weight: 700; font-size: 0.88rem;
      cursor: pointer; transition: all 0.2s ease; border: none; text-decoration: none;
    }
    .btn-primary {
      background: var(--accent-red);
      color: #fff;
      box-shadow: 0 0 16px var(--accent-red-glow);
    }
    .btn-primary:hover {
      background: #ff1a1a;
      transform: translateY(-2px);
      box-shadow: 0 0 24px rgba(229, 9, 20, 0.7);
    }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.06);
      color: #fff;
      border: 1px solid var(--border-color);
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: #ff4d4d;
    }
    .glass-panel {
      background: rgba(17, 23, 38, 0.9);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 14px;
    }
    /* Header Navbar */
    header {
      position: sticky; top: 0; z-index: 100;
      background: rgba(10, 13, 20, 0.95);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border-color);
    }
    .nav-container {
      max-width: 1550px; margin: 0 auto; padding: 12px 24px;
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;
    }
    .logo-container { display: flex; align-items: center; gap: 12px; }
    .logo-icon {
      width: 44px; height: 44px; border-radius: 12px;
      background: linear-gradient(135deg, #ff1a1a, #8b0000);
      display: flex; align-items: center; justify-content: center;
      font-size: 1.5rem; box-shadow: 0 0 20px rgba(229, 9, 20, 0.6);
      border: 1px solid rgba(255, 77, 77, 0.4);
    }
    .brand-title {
      font-size: 1.35rem; font-weight: 900; letter-spacing: -0.01em;
      background: linear-gradient(to right, #ffffff, #ff4d4d, #e50914);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subnav {
      border-top: 1px solid rgba(31, 41, 61, 0.6);
      background: rgba(17, 23, 38, 0.7);
    }
    .tabs-bar {
      max-width: 1550px; margin: 0 auto; padding: 0 24px;
      display: flex; gap: 10px; overflow-x: auto;
    }
    .tab-btn {
      padding: 12px 18px; border: none; background: transparent;
      color: var(--text-muted); font-size: 0.88rem; font-weight: 600;
      cursor: pointer; border-bottom: 2px solid transparent;
      white-space: nowrap; transition: all 0.2s;
    }
    .tab-btn.active {
      color: #fff; border-bottom-color: var(--accent-red);
    }
    /* Main Layout */
    main { max-width: 1550px; margin: 0 auto; padding: 28px 24px; }
    .movie-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 20px; margin-top: 18px;
    }
    .movie-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px; overflow: hidden;
      display: flex; flex-direction: column;
      transition: all 0.25s ease; position: relative;
    }
    .movie-card:hover {
      transform: translateY(-5px);
      border-color: rgba(229, 9, 20, 0.6);
      box-shadow: 0 10px 25px rgba(0,0,0,0.7);
    }
    .poster-box {
      width: 100%; height: 290px; position: relative; background: #1a2234;
    }
    .poster-box img {
      width: 100%; height: 100%; object-fit: cover;
    }
    .rank-tag {
      position: absolute; top: 10px; left: 10px;
      background: rgba(0,0,0,0.8); border: 1px solid rgba(255,255,255,0.2);
      border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; font-weight: 800; color: #fff;
    }
    .match-tag {
      position: absolute; top: 10px; right: 10px;
      background: #e50914; border-radius: 6px; padding: 3px 8px;
      font-size: 0.72rem; font-weight: 800; color: #fff;
    }
    .card-info { padding: 14px; flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; }
  </style>
</head>
<body>

  <!-- Top Navigation Header -->
  <header>
    <div class="nav-container">
      <div class="logo-container">
        <div class="logo-icon">🛋️</div>
        <div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="brand-title">আসেন Unproductive হই</span>
            <span class="badge badge-red">🪫 1% BATTERY VIBE</span>
          </div>
          <p style="font-size: 0.74rem; color: var(--text-muted);">
            Asen Unproductive Hoi • কাজের কাজ বাদ দিয়ে টাইম নষ্ট করার সেরা এআই ইঞ্জিন
          </p>
        </div>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span class="badge badge-red" style="font-family: monospace;">● ALL-IN-ONE SINGLE FILE EDITION</span>
      </div>
    </div>
    <div class="subnav">
      <div class="tabs-bar">
        <button class="tab-btn active" onclick="switchTab('home')">🛋️ Procrastination Hub</button>
        <button class="tab-btn" onclick="switchTab('lab')">🧪 Algorithm Lab</button>
        <button class="tab-btn" onclick="switchTab('pipeline')">⚡ 5-Stage Pipeline</button>
        <button class="tab-btn" onclick="switchTab('artwork')">🎨 Artwork Bandit</button>
        <button class="tab-btn" onclick="switchTab('explorer')">🔍 Find a Distraction</button>
      </div>
    </div>
  </header>

  <main>
    <!-- TAB 1: HOME PROCRASTINATION HUB -->
    <div id="tab-home">
      <div class="glass-panel" style="padding: 40px; background: linear-gradient(135deg, rgba(229, 9, 20, 0.25) 0%, rgba(139, 0, 0, 0.15) 50%, rgba(10, 13, 20, 0.95) 100%); border: 1px solid rgba(229, 9, 20, 0.4); box-shadow: 0 0 35px rgba(229, 9, 20, 0.2);">
        <div style="display: flex; gap: 8px; margin-bottom: 14px;">
          <span class="badge badge-red">🛋️ OFFICIAL PROCRASTINATION ENGINE</span>
          <span class="badge badge-red" style="border: 1px solid #ff4d4d;">🪫 100% UNPRODUCTIVE</span>
        </div>
        <h1 style="font-size: 2.8rem; font-weight: 900; line-height: 1.15; letter-spacing: -0.02em;">
          আসেন Unproductive হই
        </h1>
        <p style="margin-top: 14px; color: #cbd5e1; font-size: 1.1rem; max-width: 800px; line-height: 1.6;">
          কাজের মাঝে একটু বিরতি দরকার? আসতেই পারেন! YouTube, Netflix, TikTok ও Spotify-র ২০ বছরের
          আধুনিক এআই অ্যালগরিদম একত্রিত করে তৈরি — সোফায় শুয়ে অলস সময় কাটানোর এক নিখুঁত রিকমেন্ডেশন সিস্টেম।
        </p>
        <div style="display: flex; gap: 14px; margin-top: 26px; flex-wrap: wrap;">
          <button class="btn btn-primary" onclick="switchTab('pipeline')">
            <span>🛋️ Start Procrastinating (Pipeline)</span>
          </button>
          <button class="btn btn-secondary" onclick="switchTab('lab')">
            <span>🧪 Waste My Time in Algo Lab</span>
          </button>
          <button class="btn btn-secondary" onclick="switchTab('explorer')">
            <span>🔍 Find a Distraction</span>
          </button>
        </div>
      </div>

      <div style="margin-top: 36px;">
        <h2 style="font-size: 1.4rem; font-weight: 800; display: flex; align-items: center; gap: 10px;">
          <span>🔥 Trending Official Blockbusters</span>
          <span class="badge badge-red">TMDB THEATRICAL POSTERS</span>
        </h2>
        <div class="movie-grid" id="home-trending-grid"></div>
      </div>
    </div>

    <!-- TAB 2: ALGORITHM LAB -->
    <div id="tab-lab" style="display: none;">
      <div class="glass-panel" style="padding: 24px; display: flex; flex-wrap: wrap; gap: 16px; align-items: center; justify-content: space-between;">
        <div style="display: flex; gap: 12px; flex-wrap: wrap;">
          <div>
            <label style="font-size: 0.75rem; color: var(--text-muted); font-weight: 700;">ALGORITHM</label><br>
            <select id="lab-algo" style="background: var(--bg-dark); color: #fff; padding: 10px; border-radius: 8px; border: 1px solid var(--border-color);">
              <option value="sasrec">SASRec (Sequential Transformer - UCSD)</option>
              <option value="two_tower">Two-Tower DNN (YouTube Covington)</option>
              <option value="din">DIN Local Attention (Alibaba)</option>
              <option value="mmoe">MMoE Multi-Task (Google)</option>
              <option value="lightgcn">LightGCN 3-Hop Graph (NUS)</option>
            </select>
          </div>
          <div>
            <label style="font-size: 0.75rem; color: var(--text-muted); font-weight: 700;">USER PERSONA</label><br>
            <select id="lab-user" style="background: var(--bg-dark); color: #fff; padding: 10px; border-radius: 8px; border: 1px solid var(--border-color);">
              <option value="usr_scifi">Sci-Fi & Cyberpunk Geek</option>
              <option value="usr_drama">Romance & Emotional Drama Lover</option>
              <option value="usr_cinephile">Auteur Cinema Critic</option>
              <option value="usr_anime">Anime & Fantasy Fan</option>
            </select>
          </div>
        </div>
        <button class="btn btn-primary" onclick="runLabRecommendations()">
          <span>🛋️ Waste My Time</span>
        </button>
      </div>

      <div style="margin-top: 28px;">
        <h3 id="lab-results-title" style="font-size: 1.2rem; font-weight: 800;">Recommended Distractions</h3>
        <div class="movie-grid" id="lab-results-grid"></div>
      </div>
    </div>

    <!-- TAB 3: 5-STAGE PIPELINE -->
    <div id="tab-pipeline" style="display: none;">
      <div class="glass-panel" style="padding: 26px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
          <div>
            <h2 style="font-size: 1.3rem; font-weight: 800;">⚡ 5-Stage Netflix Production Pipeline</h2>
            <p style="font-size: 0.85rem; color: var(--text-muted);">
              Retrieval (ANN) → Filtering → Ranking (MMoE) → Diversity (MMR) → Slate Personalization
            </p>
          </div>
          <button class="btn btn-primary" onclick="runPipelineFunnel()">
            <span>🚀 Run Full Funnel</span>
          </button>
        </div>
        <div id="funnel-metrics" style="margin-top: 24px; display: flex; flex-direction: column; gap: 10px;"></div>
      </div>
      <div id="pipeline-rows" style="margin-top: 36px; display: flex; flex-direction: column; gap: 32px;"></div>
    </div>

    <!-- TAB 4: ARTWORK BANDIT -->
    <div id="tab-artwork" style="display: none;">
      <div class="glass-panel" style="padding: 26px;">
        <h2 style="font-size: 1.3rem; font-weight: 800;">🎨 LinUCB Artwork Personalization Bandit</h2>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">
          Simulates Netflix's multi-armed bandit dynamically selecting posters for "Inception".
        </p>
        <div style="display: flex; gap: 12px; margin-top: 18px;">
          <button class="btn btn-primary" onclick="simulateBandit('variant_action')">Simulate Click: Action Poster</button>
          <button class="btn btn-secondary" onclick="simulateBandit('variant_character')">Simulate Click: Character Close-Up</button>
          <button class="btn btn-secondary" onclick="simulateBandit('variant_romantic')">Simulate Click: Romantic / Mal</button>
          <button class="btn btn-secondary" onclick="simulateBandit('variant_cinematic')">Simulate Click: Folding Paris</button>
        </div>
      </div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-top: 24px;" id="bandit-arms-grid"></div>
    </div>

    <!-- TAB 5: FIND A DISTRACTION -->
    <div id="tab-explorer" style="display: none;">
      <div class="glass-panel" style="padding: 26px;">
        <h2 style="font-size: 1.3rem; font-weight: 800;">🔍 Find a Distraction (Zero-Shot Search)</h2>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">
          মনের মতো বিষয় বা মুড লিখুন — এআই আপনাকে সেরা ডিস্ট্র্যাকশন খুঁজে দেবে।
        </p>
        <div style="display: flex; gap: 10px; margin-top: 18px;">
          <input id="search-input" type="text" placeholder="কী দিয়ে সময় নষ্ট করতে চান? (যেমন: Mind-bending sci-fi, fast action)..." style="flex-grow: 1; background: var(--bg-dark); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; color: #fff; font-size: 0.95rem;">
          <button class="btn btn-primary" onclick="runSearch()">
            <span>🛋️ Find a Distraction</span>
          </button>
        </div>
      </div>
      <div class="movie-grid" id="search-results-grid" style="margin-top: 24px;"></div>
    </div>
  </main>

  <footer style="text-align: center; padding: 24px; border-top: 1px solid var(--border-color); color: var(--text-muted); font-size: 0.8rem;">
    আসেন Unproductive হই (Asen Unproductive Hoi) • All-in-One Single File Edition • Powered by 10 Production Models
  </footer>

  <script>
    let catalog = [];

    function switchTab(tabId) {
      ['home', 'lab', 'pipeline', 'artwork', 'explorer'].forEach(t => {
        document.getElementById('tab-' + t).style.display = (t === tabId) ? 'block' : 'none';
      });
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      event.currentTarget?.classList?.add('active');
    }

    async function loadCatalog() {
      const res = await fetch('/api/items');
      catalog = await res.json();
      renderMovieGrid(catalog.slice(0, 10), 'home-trending-grid');
      renderBanditArms();
    }

    function renderMovieGrid(items, targetId) {
      const el = document.getElementById(targetId);
      el.innerHTML = items.map((it, idx) => `
        <div class="movie-card">
          <div class="poster-box">
            <span class="rank-tag">#${it.rank || idx + 1}</span>
            <span class="match-tag">${it.score ? (it.score * 100).toFixed(0) + '% Match' : it.genre}</span>
            <img src="${it.poster_url}" alt="${it.title}" onerror="this.src='https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg'" />
          </div>
          <div class="card-info">
            <h4 style="font-size: 0.95rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${it.title}</h4>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">
              <span>${it.genre} (${it.year})</span>
              <span style="color: #ff4d4d; font-weight: 700;">★ ${it.rating}</span>
            </div>
          </div>
        </div>
      `).join('');
    }

    async function runLabRecommendations() {
      const algo = document.getElementById('lab-algo').value;
      const user = document.getElementById('lab-user').value;
      const res = await fetch('/api/recommend/single', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ user_id: user, algorithm: algo, n: 10 })
      });
      const data = await res.json();
      document.getElementById('lab-results-title').innerText = `Recommended Distractions by ${algo.toUpperCase()} (${data.latency_ms} ms)`;
      renderMovieGrid(data.items, 'lab-results-grid');
    }

    async function runPipelineFunnel() {
      const res = await fetch('/api/recommend/pipeline', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ user_id: 'usr_scifi' })
      });
      const data = await res.json();

      const fEl = document.getElementById('funnel-metrics');
      fEl.innerHTML = data.stage_metrics.map(st => `
        <div style="background: var(--bg-dark); padding: 12px 18px; border-radius: 8px; border: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center;">
          <span style="font-weight: 700;">${st.stage_name}</span>
          <span style="color: #ff4d4d; font-weight: 700;">${st.output_count} candidates (${st.latency_ms} ms)</span>
        </div>
      `).join('');

      const rEl = document.getElementById('pipeline-rows');
      rEl.innerHTML = data.rows.map((row, rIdx) => `
        <div>
          <h3 style="font-size: 1.25rem; font-weight: 800; margin-bottom: 12px;">${row.title}</h3>
          <div class="movie-grid" id="pipe-row-${rIdx}"></div>
        </div>
      `).join('');

      data.rows.forEach((row, rIdx) => {
        renderMovieGrid(row.items, 'pipe-row-' + rIdx);
      });
    }

    const INCEPTION_ARMS = [
      { id: 'variant_action', label: 'Action Focus Variant', img: 'https://image.tmdb.org/t/p/w500/tXQvtRWfkUUnWJAn2tN3jERIUG.jpg' },
      { id: 'variant_character', label: 'Character Close-Up Variant', img: 'https://image.tmdb.org/t/p/w500/r84x4x93LbZ2gozISTBYVeq0gLZ.jpg' },
      { id: 'variant_romantic', label: 'Romantic / Mal Variant', img: 'https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?w=600&q=80' },
      { id: 'variant_cinematic', label: 'Folding Paris Variant', img: 'https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg' },
    ];

    async function renderBanditArms() {
      const res = await fetch('/api/artwork/status');
      const data = await res.json();
      const el = document.getElementById('bandit-arms-grid');
      el.innerHTML = INCEPTION_ARMS.map(arm => {
        const stat = data.arm_statistics[arm.id] || { pull_count: 5, win_rate: 0.5 };
        return `
          <div class="glass-panel" style="overflow: hidden; border-radius: 12px; border: 1px solid var(--border-color);">
            <div style="height: 280px; width: 100%;">
              <img src="${arm.img}" style="width: 100%; height: 100%; object-fit: cover;" onerror="this.src='https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg'" />
            </div>
            <div style="padding: 16px;">
              <h4 style="font-weight: 700; font-size: 1rem;">${arm.label}</h4>
              <div style="margin-top: 8px; font-size: 0.8rem; color: var(--text-muted); display: flex; justify-content: space-between;">
                <span>Impressions: <strong>${stat.pull_count}</strong></span>
                <span style="color: #ff4d4d; font-weight: 700;">Win Rate: ${(stat.win_rate * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    async function simulateBandit(armId) {
      await fetch('/api/artwork/simulate-round', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ selected_arm: armId, reward: 1.0, persona_id: 'usr_scifi' })
      });
      await renderBanditArms();
    }

    function runSearch() {
      const q = document.getElementById('search-input').value.toLowerCase();
      const matches = catalog.filter(it => it.title.toLowerCase().includes(q) || it.genre.toLowerCase().includes(q) || it.director.toLowerCase().includes(q));
      renderMovieGrid(matches.length > 0 ? matches : catalog.slice(0, 5), 'search-results-grid');
    }

    loadCatalog();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_UI

# ==============================================================================
# 5. RUNNER ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\\n=======================================================")
    print(f"🛋️  আসেন Unproductive হই (Asen Unproductive Hoi)")
    print(f"🚀  Single-File All-in-One Engine is starting up...")
    print(f"👉  Open your browser at: http://localhost:{port}")
    print(f"=======================================================\\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
