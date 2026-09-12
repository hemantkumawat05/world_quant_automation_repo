"""The live telemetry socket.

One connection carries every topic. Server-to-client only — commands go over REST — so
the client needs no request/response correlation, just a topic switch. Supports multi-tenant tokens.
"""

from __future__ import annotations

import asyncio
import contextlib

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..brain.auth import SessionInfo
from ..engine.tracker import serialise
from ..realtime import TOPIC_SESSION, TOPIC_SIMULATIONS
from ..security.jwt import verify_access_token
from ..state import AppState

log = structlog.get_logger(__name__)

router = APIRouter(tags=["realtime"])

#: Nudge the socket periodically so a dead connection is noticed rather than lingering.
HEARTBEAT_SECONDS = 25.0


@router.websocket("/ws")
async def telemetry(websocket: WebSocket) -> None:
    state: AppState = websocket.app.state.harness
    token = websocket.query_params.get("token")

    user_id: str | None = None
    if token:
        try:
            payload = verify_access_token(token, state.vault.key)
            user_id = payload.get("sub")
        except Exception:
            log.debug("ws.auth_failed")

    await state.hub.connect(websocket, user_id=user_id)

    try:
        # Determine session and active simulations for this specific user
        if user_id:
            user_session, _ = await state.auth.get_user_context(user_id)
            active_sims = await state.tracker.active_for_user(user_id)
        else:
            user_session = SessionInfo.anonymous()
            active_sims = []

        # Send current state immediately so a newly opened tab is not blank
        await websocket.send_json({"topic": TOPIC_SESSION, "payload": user_session.to_dict()})
        await websocket.send_json(
            {
                "topic": TOPIC_SIMULATIONS,
                "payload": [serialise(r) for r in active_sims],
            }
        )

        while True:
            try:
                # Any inbound frame is treated as a keepalive; there are no commands.
                await asyncio.wait_for(websocket.receive_text(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                await websocket.send_json({"topic": "ping", "payload": None})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.debug("ws.closed_unexpectedly", exc_info=True)
    finally:
        await state.hub.disconnect(websocket)
        with contextlib.suppress(Exception):
            await websocket.close()
