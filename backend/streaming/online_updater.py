"""
Online Dynamic Updater
Listens to real-time events on EventBus:
- Updates online feature store
- Updates LinUCB Contextual Bandit weights when a user clicks/converts on an artwork thumbnail
- Triggers online session drift
"""

import numpy as np
from backend.streaming.event_bus import event_bus, UserInteractionEvent, EventType
from backend.streaming.feature_store import feature_store
from backend.bandits.linucb import LinUCBBandit


class OnlineUpdater:
    def __init__(self, artwork_bandit: LinUCBBandit):
        self.artwork_bandit = artwork_bandit
        # Register listener on event bus
        event_bus.subscribe(self.handle_event)

    def handle_event(self, event: UserInteractionEvent):
        # 1. Update online feature store
        item_genre = event.context.get("genre", "General")
        feature_store.update_from_event(event, item_genre=item_genre)

        # 2. If interaction is on an artwork variant, update LinUCB bandit
        if event.artwork_variant_id and event.event_type in [EventType.CLICK, EventType.WATCH_COMPLETE, EventType.LIKE]:
            reward = 1.0 if event.event_type in [EventType.WATCH_COMPLETE, EventType.LIKE] else 0.5
            genres = ["Action", "Sci-Fi", "Drama", "Comedy"]
            context_vec = feature_store.get_user_genre_vector(event.user_id, genres)
            self.artwork_bandit.update(
                selected_arm=event.artwork_variant_id,
                context_vec=context_vec,
                reward=reward,
            )
