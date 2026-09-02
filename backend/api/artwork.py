"""
Artwork Personalization Bandit Endpoints (Netflix Style)
Allows the frontend to inspect LinUCB arm distributions (Expected Rewards, Confidence Bounds)
and simulate rounds for different taste profiles (e.g. Action fan vs Romance fan)
to watch the bandit explore and converge in real-time.
"""

import numpy as np
from typing import Dict, List, Any
from pydantic import BaseModel
from fastapi import APIRouter
from backend.api.recommendations import SHARED_ARTWORK_BANDIT
from backend.bandits.linucb import BanditDecision

router = APIRouter(prefix="/api/artwork", tags=["Artwork Personalization"])


class TasteProfile(BaseModel):
    name: str
    action_affinity: float
    scifi_affinity: float
    romance_affinity: float
    cinematic_affinity: float


class SimulateRoundRequest(BaseModel):
    item_id: str = "mov_0001"
    taste_profile: TasteProfile
    user_clicked: bool = True


@router.get("/status")
def get_bandit_status() -> Dict[str, Any]:
    stats = {}
    for arm in SHARED_ARTWORK_BANDIT.arm_names:
        pulls = SHARED_ARTWORK_BANDIT.pull_counts[arm]
        rewards = SHARED_ARTWORK_BANDIT.reward_sums[arm]
        stats[arm] = {
            "pull_count": pulls,
            "total_rewards": round(rewards, 2),
            "win_rate": round(rewards / max(1, pulls), 3),
        }
    return {
        "arms": SHARED_ARTWORK_BANDIT.arm_names,
        "alpha_exploration": SHARED_ARTWORK_BANDIT.alpha,
        "arm_statistics": stats,
    }


@router.post("/simulate-round")
def simulate_bandit_round(req: SimulateRoundRequest) -> Dict[str, Any]:
    tp = req.taste_profile
    context_vec = np.array(
        [tp.action_affinity, tp.scifi_affinity, tp.romance_affinity, tp.cinematic_affinity],
        dtype=np.float32,
    )

    # 1. Bandit decision
    decision = SHARED_ARTWORK_BANDIT.select_arm(context_vec, item_id=req.item_id)
    selected_arm = decision.selected_arm

    # 2. Compute simulated reward based on persona affinity matching
    affinity_map = {
        "variant_action": tp.action_affinity,
        "variant_character": tp.scifi_affinity,
        "variant_romantic": tp.romance_affinity,
        "variant_cinematic": tp.cinematic_affinity,
    }

    match_score = affinity_map.get(selected_arm, 0.5)
    reward = 1.0 if req.user_clicked and (np.random.random() < match_score + 0.2) else 0.0

    # 3. Update bandit
    SHARED_ARTWORK_BANDIT.update(selected_arm=selected_arm, context_vec=context_vec, reward=reward)

    return {
        "decision": decision.model_dump(),
        "reward_received": reward,
        "updated_pull_count": SHARED_ARTWORK_BANDIT.pull_counts[selected_arm],
    }
