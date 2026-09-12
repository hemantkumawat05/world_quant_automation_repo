"""Sign-in, session state, and cached platform metadata."""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ..brain.auth import SessionInfo
from ..brain.settings_schema import resolve_options, validate_settings
from ..realtime import TOPIC_SESSION
from .deps import OptionalUser, State, User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Omit both fields to sign in with the stored credential."""

    email: str | None = Field(default=None, description="Leave empty to use the saved login")
    password: str | None = Field(default=None, repr=False)


@router.get("/status")
async def status(
    state: State,
    user: OptionalUser,
    refresh: bool = Query(
        False, description="Re-validate against BRAIN instead of returning cached state"
    ),
) -> dict[str, Any]:
    """Current session for the requesting user."""
    if user:
        info = await state.auth.status(user_id=user.user_id, refresh=refresh)
        return {
            **info.to_dict(),
            "storedEmail": user.email or await state.auth.stored_email(user_id=user.user_id),
        }
    return {
        **SessionInfo.anonymous().to_dict(),
        "storedEmail": None,
    }


@router.post("/login")
async def login(payload: LoginRequest, state: State) -> dict[str, Any]:
    """Sign in to BRAIN.

    Solves the ALTCHA proof-of-work, exchanges Basic auth for a session cookie,
    and returns an access token.
    """
    info, token = await state.auth.login(payload.email, payload.password)

    if info.authenticated:
        state.engine.configure_from_permissions(info.permissions)
        with contextlib.suppress(Exception):
            await state.auth.refresh_metadata()

    if info.user_id:
        await state.hub.broadcast(TOPIC_SESSION, info.to_dict(), user_id=info.user_id)
    else:
        await state.hub.broadcast(TOPIC_SESSION, info.to_dict())

    return {
        **info.to_dict(),
        "token": token,
    }


@router.post("/logout")
async def logout(state: State, user: OptionalUser) -> dict[str, Any]:
    user_id = user.user_id if user else None
    await state.auth.logout(user_id=user_id)
    info = SessionInfo.anonymous()
    if user_id:
        await state.hub.broadcast(TOPIC_SESSION, info.to_dict(), user_id=user_id)
    else:
        await state.hub.broadcast(TOPIC_SESSION, info.to_dict())
    return info.to_dict()


@router.delete("/credential")
async def forget_credential(state: State, user: OptionalUser) -> dict[str, bool]:
    """Erase the stored login and cached session from local storage."""
    user_id = user.user_id if user else None
    await state.auth.forget(user_id=user_id)
    if user_id:
        await state.hub.broadcast(TOPIC_SESSION, SessionInfo.anonymous().to_dict(), user_id=user_id)
    else:
        await state.hub.broadcast(TOPIC_SESSION, SessionInfo.anonymous().to_dict())
    return {"ok": True}


@router.get("/settings-schema")
async def settings_schema(
    state: State,
    refresh: bool = Query(False, description="Re-fetch from BRAIN"),
) -> dict[str, Any]:
    """Valid simulation settings, from ``OPTIONS /simulations``."""
    if refresh:
        return {"cached": False, "schema": await state.auth.refresh_metadata()}
    cached = await state.auth.cached_settings_schema()
    if cached is None:
        return {"cached": False, "schema": await state.auth.refresh_metadata()}
    return {"cached": True, "schema": cached}


class ResolveOptionsRequest(BaseModel):
    """Settings chosen so far. Partial is fine — that is the point."""

    settings: dict[str, Any] = Field(default_factory=dict)


@router.post("/settings-options")
async def settings_options(payload: ResolveOptionsRequest, state: State) -> dict[str, Any]:
    schema = await state.auth.cached_settings_schema()
    if schema is None:
        schema = await state.auth.refresh_metadata()

    return {
        "fields": resolve_options(schema, payload.settings),
        "problems": validate_settings(schema, payload.settings),
        "missing": validate_settings(schema, payload.settings, require_all=True),
    }


@router.get("/operators")
async def operators(
    state: State,
    refresh: bool = Query(False, description="Re-fetch from BRAIN"),
) -> dict[str, Any]:
    """The Fast Expression operator reference, for autocomplete and validation."""
    if not refresh:
        cached = await state.auth.cached_operators()
        if cached is not None:
            return {"cached": True, "count": len(cached), "operators": cached}
    fresh = await state.auth.refresh_operators()
    return {"cached": False, "count": len(fresh), "operators": fresh}
