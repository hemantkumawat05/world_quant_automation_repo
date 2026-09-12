"""Simulations and their resulting alphas."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..brain.schemas import SimulationRequest, SimulationSettings
from ..brain.settings_schema import validate_settings
from ..engine.tracker import serialise
from .deps import OptionalUser, State, User

router = APIRouter(prefix="/api", tags=["simulations"])


class SubmitRequest(BaseModel):
    """Start one simulation."""

    expression: str = Field(description="Fast Expression code")
    settings: SimulationSettings
    type: str = "REGULAR"
    task: str = Field(default="manual", description="Groups work for quotas and filtering")


@router.post("/simulations")
async def submit(payload: SubmitRequest, state: State, user: User) -> dict[str, Any]:
    """Submit a simulation and begin tracking it."""
    if state.engine.daily_limit_hit:
        raise HTTPException(
            429,
            detail={
                "code": "daily_limit_reached",
                "message": "Daily simulation limit reached. It resets at midnight US Eastern.",
            },
        )
    schema = await state.auth.cached_settings_schema()
    if schema:
        problems = validate_settings(
            schema, payload.settings.model_dump(by_alias=True, exclude_none=True)
        )
        if problems:
            raise HTTPException(
                422,
                detail={
                    "code": "invalid_settings",
                    "message": " ".join(problems),
                    "problems": problems,
                },
            )

    request = SimulationRequest(
        type=payload.type,  # type: ignore[arg-type]
        settings=payload.settings,
        regular=payload.expression,
    )
    record = await state.tracker.submit(
        request,
        task=payload.task,
        user_id=user.user_id,
        endpoints=user.endpoints,
    )
    return serialise(record)


class QueueRequest(BaseModel):
    """Queue a set of simulations for the batch engine to pack and run."""

    simulations: list[SubmitRequest] = Field(
        description="Each carries its own expression and settings"
    )
    task: str = Field(default="manual", description="Groups work for quotas and filtering")
    skip_duplicates: bool = Field(
        default=True,
        description="Skip anything already simulated, so a repeat costs no daily quota",
    )


@router.post("/simulations/queue")
async def enqueue(payload: QueueRequest, state: State) -> dict[str, Any]:
    """Hand work to the batch engine.

    Nothing is sent immediately. The engine groups queued simulations by the five
    fields a multi-simulation's children must share, then keeps the eight concurrent
    slots full. Anything already simulated is skipped rather than re-run.
    """
    schema = await state.auth.cached_settings_schema()
    requests: list[SimulationRequest] = []

    for index, item in enumerate(payload.simulations):
        settings = item.settings.model_dump(by_alias=True, exclude_none=True)
        if schema:
            problems = validate_settings(schema, settings)
            if problems:
                raise HTTPException(
                    422,
                    detail={
                        "code": "invalid_settings",
                        "message": f"Simulation {index + 1}: {' '.join(problems)}",
                        "index": index,
                        "problems": problems,
                    },
                )
        requests.append(
            SimulationRequest(
                type=item.type,  # type: ignore[arg-type]
                settings=item.settings,
                regular=item.expression,
            )
        )

    result = await state.engine.enqueue(
        requests, task=payload.task, skip_duplicates=payload.skip_duplicates
    )
    return {**result, "status": await state.engine.status()}


@router.get("/simulations/engine")
async def engine_status(state: State) -> dict[str, Any]:
    """Slot occupancy, queue depth and task quotas — what the matrix header shows."""
    return await state.engine.status()


@router.post("/simulations/engine/tick")
async def engine_tick(state: State) -> dict[str, Any]:
    """Run one scheduling round now instead of waiting for the next tick."""
    submitted = await state.engine.tick()
    return {"submitted": submitted, "status": await state.engine.status()}


@router.delete("/simulations/queue")
async def drop_queue(state: State, task: str | None = None) -> dict[str, int]:
    """Discard queued work that has not been submitted yet.

    Safe by construction: a queued row has no platform id because nothing was sent.
    """
    return {"dropped": await state.engine.drop_queued(task)}


class QuotaRequest(BaseModel):
    name: str
    max_slots: int = Field(ge=0, le=8)
    enabled: bool = True


@router.put("/simulations/quotas")
async def set_quota(payload: QuotaRequest, state: State) -> dict[str, Any]:
    """Cap how many concurrent slots a named task may hold.

    Without a cap one long sweep starves everything else.
    """
    await state.engine.set_quota(payload.name, payload.max_slots, enabled=payload.enabled)
    return await state.engine.status()


@router.get("/simulations/active")
async def active(state: State, user: OptionalUser) -> list[dict[str, Any]]:
    """Everything currently pending or running — what the matrix renders."""
    user_id = user.user_id if user else None
    return [serialise(r) for r in await state.tracker.active(user_id=user_id)]


@router.get("/simulations/recent")
async def recent(
    state: State,
    user: OptionalUser,
    limit: int = Query(100, ge=1, le=500),
) -> list[dict[str, Any]]:
    user_id = user.user_id if user else None
    return [serialise(r) for r in await state.tracker.recent(limit, user_id=user_id)]


@router.get("/simulations/quota")
async def quota(state: State) -> dict[str, Any]:
    """Daily simulation quota as last reported by the platform.

    The daily cap binds long before the concurrency limit does, so this is shown
    prominently rather than buried.
    """
    snapshot = await state.tracker.latest_quota()
    if snapshot is None:
        return {"known": False}
    reset = snapshot.reset_seconds
    if reset is not None and snapshot.observed_at is not None:
        # The header counts down from when it was observed, not from now.
        observed = snapshot.observed_at
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=UTC)
        reset = max(0.0, reset - (datetime.now(UTC) - observed).total_seconds())
    return {
        "known": True,
        "limit": snapshot.limit_total,
        "remaining": snapshot.remaining,
        "resetSeconds": reset,
        "observedAt": snapshot.observed_at.isoformat() if snapshot.observed_at else None,
    }


@router.get("/simulations/{record_id}")
async def get_simulation(record_id: int, state: State) -> dict[str, Any]:
    record = await state.tracker.get(record_id)
    if record is None:
        raise HTTPException(404, "No such simulation")
    return serialise(record)


@router.post("/simulations/{record_id}/cancel")
async def cancel(record_id: int, state: State) -> dict[str, Any]:
    """Cancel a running simulation.

    Uses the stored platform id — the reason that id is written to disk before the
    submission request is even sent.
    """
    record = await state.tracker.get(record_id)
    if record is None:
        raise HTTPException(404, "No such simulation")

    acknowledged = await state.tracker.cancel(record_id)
    updated = await state.tracker.get(record_id)
    return {
        "acknowledged": acknowledged,
        "simulation": serialise(updated) if updated else None,
    }
