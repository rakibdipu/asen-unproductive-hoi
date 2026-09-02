"""
Real-Time Event Ingestion & WebSocket Stream Endpoints
Allows frontends and mobile simulators to send clicks, watch completions, skips, and likes.
Streams live ticker updates to subscribed WebSocket clients.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.streaming.event_bus import event_bus, UserInteractionEvent

router = APIRouter(prefix="/api/events", tags=["Events & Streaming"])


@router.post("")
async def ingest_event(event: UserInteractionEvent) -> Dict[str, Any]:
    await event_bus.publish(event)
    return {"status": "success", "event_id": event.event_id, "timestamp": event.timestamp}


@router.get("/recent")
def get_recent_events(limit: int = 30) -> List[UserInteractionEvent]:
    return event_bus.get_recent_events(limit=limit)


@router.websocket("/ws")
async def websocket_event_stream(websocket: WebSocket):
    await event_bus.register_websocket(websocket)
    try:
        while True:
            # Keep socket alive and receive client ping/pong or actions
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        event_bus.remove_websocket(websocket)
    except Exception:
        event_bus.remove_websocket(websocket)
