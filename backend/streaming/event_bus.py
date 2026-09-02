"""
Real-Time Event Broker & WebSocket Stream (Kafka / Flink Simulation)
Ingests live user interaction events (views, clicks, watches, likes, skips)
and broadcasts them to connected WebSocket UI clients and online feature store.
"""

import asyncio
import time
from enum import Enum
from typing import Dict, List, Any, Set, Callable, Optional
from pydantic import BaseModel, Field
from fastapi import WebSocket


class EventType(str, Enum):
    IMPRESSION = "impression"
    CLICK = "click"
    WATCH_PARTIAL = "watch_partial"
    WATCH_COMPLETE = "watch_complete"
    SKIP = "skip"
    LIKE = "like"
    DISLIKE = "dislike"
    SHARE = "share"


class UserInteractionEvent(BaseModel):
    event_id: str
    user_id: str
    item_id: str
    event_type: EventType
    timestamp: float = Field(default_factory=time.time)
    watch_ratio: float = 0.0
    rating: Optional[float] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    artwork_variant_id: Optional[str] = None


class EventBus:
    """Async Pub-Sub Event Bus for real-time user streaming."""

    def __init__(self):
        self.active_websockets: Set[WebSocket] = set()
        self.subscribers: List[Callable[[UserInteractionEvent], None]] = []
        self.event_history: List[UserInteractionEvent] = []
        self.max_history: int = 200

    async def register_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.add(websocket)

    def remove_websocket(self, websocket: WebSocket):
        self.active_websockets.discard(websocket)

    def subscribe(self, callback: Callable[[UserInteractionEvent], None]):
        self.subscribers.append(callback)

    async def publish(self, event: UserInteractionEvent):
        # Add to rolling history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        # Notify internal python subscribers
        for sub in self.subscribers:
            try:
                sub(event)
            except Exception:
                pass

        # Broadcast to all connected WebSockets
        if self.active_websockets:
            payload = event.model_dump_json()
            disconnected = set()
            for ws in self.active_websockets:
                try:
                    await ws.send_text(payload)
                except Exception:
                    disconnected.add(ws)
            for dead_ws in disconnected:
                self.active_websockets.discard(dead_ws)

    def get_recent_events(self, limit: int = 50) -> List[UserInteractionEvent]:
        return self.event_history[-limit:]


event_bus = EventBus()
