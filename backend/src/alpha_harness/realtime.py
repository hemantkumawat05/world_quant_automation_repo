"""WebSocket fan-out for live telemetry.

Supports multi-tenant isolated channels: messages tagged with a user_id are only
dispatched to WebSocket connections belonging to that user.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import structlog
from fastapi import WebSocket

log = structlog.get_logger(__name__)


class Hub:
    """Tracks connected clients and broadcasts messages to them."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._user_clients: dict[str, set[WebSocket]] = {}
        self._socket_user: dict[WebSocket, str] = {}
        self._latest_global: dict[str, dict[str, Any]] = {}
        self._latest_user: dict[str, dict[str, dict[str, Any]]] = {}

    async def connect(self, websocket: WebSocket, user_id: str | None = None) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        if user_id:
            self._socket_user[websocket] = user_id
            self._user_clients.setdefault(user_id, set()).add(websocket)

        # Replay global latest
        for message in list(self._latest_global.values()):
            with contextlib.suppress(Exception):
                await websocket.send_text(json.dumps(message, default=str))

        # Replay user latest
        if user_id and user_id in self._latest_user:
            for message in list(self._latest_user[user_id].values()):
                with contextlib.suppress(Exception):
                    await websocket.send_text(json.dumps(message, default=str))

        log.debug("hub.connected", clients=len(self._clients), user_id=user_id)

    async def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)
        user_id = self._socket_user.pop(websocket, None)
        if user_id and user_id in self._user_clients:
            self._user_clients[user_id].discard(websocket)
            if not self._user_clients[user_id]:
                self._user_clients.pop(user_id, None)
        log.debug("hub.disconnected", clients=len(self._clients), user_id=user_id)

    async def broadcast(self, topic: str, payload: Any, user_id: str | None = None) -> None:
        """Send message to relevant clients."""
        message = {"topic": topic, "payload": payload}
        text = json.dumps(message, default=str)

        if user_id:
            self._latest_user.setdefault(user_id, {})[topic] = message
            target_clients = list(self._user_clients.get(user_id, set()))
        else:
            self._latest_global[topic] = message
            target_clients = list(self._clients)

        if not target_clients:
            return

        dead: list[WebSocket] = []
        for client in target_clients:
            try:
                await client.send_text(text)
            except Exception:
                dead.append(client)

        for client in dead:
            await self.disconnect(client)

    @property
    def client_count(self) -> int:
        return len(self._clients)


#: Topic names shared with the frontend. Keep in sync with frontend/src/lib/realtime.ts.
TOPIC_SIMULATIONS = "simulations"
TOPIC_SYNC = "sync"
TOPIC_SESSION = "session"
TOPIC_STUDIES = "studies"
TOPIC_TASKS = "tasks"
