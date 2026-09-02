"""
Real-Time Online Feature Store (Redis / Feast Simulation)
Stores low-latency serving features:
- Real-time user session history (last 50 items)
- Dynamic genre engagement weights
- Item impression counters (for novelty and fatigue penalty)
- Last active timestamp and device context
"""

import time
import numpy as np
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from backend.streaming.event_bus import UserInteractionEvent, EventType


class OnlineFeatureStore:
    def __init__(self):
        # user_id -> list of recent item_ids
        self.user_session_items: Dict[str, List[str]] = defaultdict(list)
        # user_id -> genre preference counter {genre: weight}
        self.user_genre_affinity: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # item_id -> total impression count
        self.item_impression_counts: Dict[str, int] = defaultdict(int)
        # user_id -> last active timestamp
        self.user_last_active: Dict[str, float] = {}

    def update_from_event(self, event: UserInteractionEvent, item_genre: str = "General"):
        uid = event.user_id
        iid = event.item_id
        self.user_last_active[uid] = event.timestamp

        # Add to session items
        if iid not in self.user_session_items[uid]:
            self.user_session_items[uid].append(iid)
            if len(self.user_session_items[uid]) > 50:
                self.user_session_items[uid].pop(0)

        # Track impression count
        if event.event_type == EventType.IMPRESSION:
            self.item_impression_counts[iid] += 1

        # Dynamic genre preference reinforcement
        weight_delta = 0.0
        if event.event_type == EventType.LIKE:
            weight_delta = 1.0
        elif event.event_type == EventType.WATCH_COMPLETE:
            weight_delta = 0.8
        elif event.event_type == EventType.CLICK:
            weight_delta = 0.3
        elif event.event_type == EventType.SKIP:
            weight_delta = -0.4

        if weight_delta != 0.0:
            current = self.user_genre_affinity[uid][item_genre]
            self.user_genre_affinity[uid][item_genre] = max(0.0, current + weight_delta)

    def get_user_session_history(self, user_id: str) -> List[str]:
        return self.user_session_items.get(user_id, [])

    def get_user_genre_vector(self, user_id: str, genre_list: List[str]) -> np.ndarray:
        aff = self.user_genre_affinity.get(user_id, {})
        vec = np.array([aff.get(g, 0.1) for g in genre_list], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec = np.full(len(genre_list), 1.0 / np.sqrt(len(genre_list)), dtype=np.float32)
        return vec

    def get_item_impressions(self, item_id: str) -> int:
        return self.item_impression_counts.get(item_id, 0)


feature_store = OnlineFeatureStore()
