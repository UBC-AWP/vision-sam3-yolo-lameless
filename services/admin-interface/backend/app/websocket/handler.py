"""
WebSocket Manager for real-time updates
Handles connections for pipeline status, health, queue, and rater updates
"""
import asyncio
import json
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for different channels.

    Channels:
    - pipeline: Pipeline status updates
    - health: System health updates
    - queue: Processing queue updates
    - rater: Rater activity updates
    """

    def __init__(self):
        # Map of channel -> set of connections
        self.connections: Dict[str, Set[WebSocket]] = {
            "pipeline": set(),
            "health": set(),
            "queue": set(),
            "rater": set()
        }
        # User ID -> WebSocket mapping for targeted messages
        self.user_connections: Dict[str, Set[WebSocket]] = {}
        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str, user_id: Optional[str] = None):
        """
        Accept a new WebSocket connection and add to channel.
        """
        await websocket.accept()

        async with self._lock:
            if channel not in self.connections:
                self.connections[channel] = set()
            self.connections[channel].add(websocket)

            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(websocket)

        logger.info(f"WebSocket connected to channel: {channel}")

    async def disconnect(self, websocket: WebSocket, channel: str, user_id: Optional[str] = None):
        """
        Remove a WebSocket connection from channel.
        """
        async with self._lock:
            if channel in self.connections:
                self.connections[channel].discard(websocket)

            if user_id and user_id in self.user_connections:
                self.user_connections[user_id].discard(websocket)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]

        logger.info(f"WebSocket disconnected from channel: {channel}")

    async def broadcast(self, channel: str, message: dict):
        """
        Broadcast a message to all connections in a channel.
        """
        if channel not in self.connections:
            return

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        message_json = json.dumps(message)

        # Get connections safely
        async with self._lock:
            connections = list(self.connections.get(channel, set()))

        # Send to all connections
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send message: {e}")
                disconnected.append(websocket)

        # Clean up disconnected sockets
        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    self.connections[channel].discard(ws)

    async def send_to_user(self, user_id: str, message: dict):
        """
        Send a message to a specific user's connections.
        """
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        message_json = json.dumps(message)

        async with self._lock:
            connections = list(self.user_connections.get(user_id, set()))

        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_text(message_json)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected sockets
        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    self.user_connections[user_id].discard(ws)

    async def broadcast_pipeline_status(self, service_name: str, status: str, details: dict = None):
        """
        Broadcast pipeline status update.
        """
        await self.broadcast("pipeline", {
            "type": "pipeline_status",
            "service": service_name,
            "status": status,
            "details": details or {}
        })

    async def broadcast_health_update(self, component: str, status: str, metrics: dict = None):
        """
        Broadcast system health update.
        """
        await self.broadcast("health", {
            "type": "health_update",
            "component": component,
            "status": status,
            "metrics": metrics or {}
        })

    async def broadcast_queue_update(self, job_id: str, status: str, progress: float = 0.0, **kwargs):
        """
        Broadcast queue job update.
        """
        await self.broadcast("queue", {
            "type": "queue_update",
            "job_id": job_id,
            "status": status,
            "progress": progress,
            **kwargs
        })

    async def broadcast_rater_update(self, event_type: str, data: dict):
        """
        Broadcast rater activity update.
        """
        await self.broadcast("rater", {
            "type": "rater_update",
            "event": event_type,
            "data": data
        })

    def get_connection_count(self, channel: str = None) -> int:
        """
        Get the number of active connections.
        """
        if channel:
            return len(self.connections.get(channel, set()))
        return sum(len(conns) for conns in self.connections.values())


# Global WebSocket manager instance
ws_manager = WebSocketManager()


# ============== WEBSOCKET ENDPOINT HANDLERS ==============

async def websocket_endpoint(websocket: WebSocket, channel: str, user_id: Optional[str] = None):
    """
    Generic WebSocket endpoint handler.
    """
    await ws_manager.connect(websocket, channel, user_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle ping/pong for keepalive
                if data == "ping":
                    await websocket.send_text("pong")
                else:
                    # Handle other messages (e.g., subscriptions)
                    message = json.loads(data)
                    if message.get("type") == "subscribe":
                        # Could add additional channel subscriptions here
                        pass
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket, channel, user_id)
