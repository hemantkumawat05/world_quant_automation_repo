"""The assistant: keys, models, prompts, and the three things it can do.

``GET /api/llm/prompts`` returns every system prompt in full. That is deliberate — the
prompt decides what an answer looks like and is otherwise invisible, so it is served
rather than hidden. Nothing here is a secret; the keys are, and those only ever leave as
a masked hint.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ..catalog.queries import Tuple4
from ..llm.keys import serialise
from ..llm.prompts import PROMPTS
from .deps import State

router = APIRouter(prefix="/api/llm", tags=["assistant"])


class Scope(BaseModel):
    instrument_type: str = "EQUITY"
    region: str
    delay: int
    universe: str

    def to_tuple(self) -> Tuple4:
        return Tuple4(
            instrument_type=self.instrument_type,
            region=self.region,
            delay=self.delay,
            universe=self.universe,
        )


# --- setup ----------------------------------------------------------------


@router.get("/models")
async def models(state: State) -> dict[str, Any]:
    """The model roster with each one's daily budget.

    Requests-per-day varies twenty-five-fold across these models and is the limit that
    ends a session, so it is returned with every entry rather than hidden in a help page.
    """
    return state.llm.registry.to_dict()


# --- prompts --------------------------------------------------------------


@router.get("/prompts")
async def list_prompts() -> dict[str, Any]:
    """Every prompt, in full.

    Served rather than hidden: a prompt decides what an answer looks like and is
    otherwise invisible in the output.
    """
    return {
        "prompts": [
            {
                "slug": p.slug,
                "label": p.label,
                "purpose": p.purpose,
                "context": p.context,
                "model": p.model,
                "temperature": p.temperature,
                "body": p.body,
                "characters": len(p.body),
                # Roughly four characters to a token. Worth showing: prompt tokens come
                # out of the same per-minute budget as the answer.
                "estimatedTokens": max(1, len(p.body) // 4),
            }
            for p in PROMPTS.values()
        ],
    }


class RunRequest(BaseModel):
    """Run one of the shipped prompts."""

    prompt: str = Field(description="A prompt slug")
    input: str = Field(description="Your question, idea, or the thing to work on")
    scope: Scope | None = Field(
        default=None, description="Required when the prompt asks for catalog data"
    )
    dataset_ids: list[str] = Field(default_factory=list)
    model: str | None = None


@router.post("/run")
async def run_prompt(body: RunRequest, state: State) -> dict[str, Any]:
    """The one path every prompt goes through."""
    answer = await state.llm.run(
        body.prompt,
        body.input,
        scope=body.scope.to_tuple() if body.scope else None,
        dataset_ids=body.dataset_ids,
        model_id=body.model,
    )
    return answer.to_dict()


@router.get("/keys")
async def list_keys(state: State, user: OptionalUser) -> dict[str, Any]:
    """Keys, today's usage, and how much budget is left across all of them."""
    user_id = user.user_id if user else None
    return await state.llm.keys.status(state.llm.registry, user_id=user_id)


@router.get("/providers")
async def providers() -> dict[str, Any]:
    """Every assistant that can answer, and how to get a free key for it."""
    from ..llm.providers import catalogue

    return catalogue()


class AddKey(BaseModel):
    key: str = Field(description="An assistant API key. Sealed at rest; never returned.")
    label: str | None = Field(default=None, description="Which account this key belongs to")
    provider: str = Field(default="google", description="Whose key this is")


@router.post("/keys", status_code=201)
async def add_key(body: AddKey, state: State, user: OptionalUser) -> dict[str, Any]:
    user_id = user.user_id if user else None
    row = await state.llm.keys.add(body.key, body.label, provider=body.provider, user_id=user_id)
    return serialise(row)


@router.post("/keys/{key_id}/check")
async def check_key(key_id: int, state: State) -> dict[str, Any]:
    """Confirm a key works. Costs nothing against the generation quota."""
    return await state.llm.check_key(key_id)


@router.post("/keys/check")
async def check_all_keys(state: State) -> list[dict[str, Any]]:
    return await state.llm.check_all()


class KeyToggle(BaseModel):
    enabled: bool


@router.put("/keys/{key_id}")
async def toggle_key(key_id: int, body: KeyToggle, state: State) -> dict[str, Any]:
    return serialise(await state.llm.keys.set_enabled(key_id, body.enabled))


@router.delete("/keys/{key_id}", status_code=204)
async def remove_key(key_id: int, state: State) -> None:
    await state.llm.keys.remove(key_id)
    state.llm.forget(key_id)


# --- context --------------------------------------------------------------


@router.get("/context")
async def context(
    state: State,
    region: str,
    delay: int,
    universe: str,
    instrument_type: str = "EQUITY",
    rendered: bool = Query(False, description="Return the exact text the model receives"),
) -> dict[str, Any]:
    """Exactly what the model is shown about your data.

    The hierarchy with metadata, and no individual fields — tens of thousands of field
    names would fill the context window and leave no room to think. Set ``rendered`` to
    read the literal text, so nothing about the assistant is a black box.
    """
    scope = Scope(
        instrument_type=instrument_type, region=region, delay=delay, universe=universe
    ).to_tuple()
    if rendered:
        text, meta = await state.llm.context.render(scope)
        return {"text": text, **meta}
    return await state.llm.context.tree(scope)


# --- the three features ---------------------------------------------------


class AdviseRequest(BaseModel):
    question: str = Field(description="An economic idea, intuition, or question")
    scope: Scope
    model: str | None = None


@router.post("/advise")
async def advise(body: AdviseRequest, state: State) -> dict[str, Any]:
    """Which datasets could implement this idea, and what could go wrong with it."""
    answer = await state.llm.advise(body.question, body.scope.to_tuple(), model_id=body.model)
    return answer.to_dict()


class PowerPoolRequest(BaseModel):
    """Generate finished Power Pool expressions, then check every one."""

    scope: Scope
    dataset_ids: list[str] = Field(
        default_factory=list,
        description=("Narrow to these datasets so their fields are shown, not the whole tree"),
    )
    count: int = Field(default=30, ge=1, le=100)
    brief: str = Field(default="", description="What you are looking for, in your own words")
    model: str | None = None
    use_seeds: bool = Field(
        default=True,
        description="Show it your best alphas in this scope so it builds on what already works",
    )


@router.post("/power-pool")
async def power_pool(body: PowerPoolRequest, state: State) -> dict[str, Any]:
    """Ask for a batch of Power Pool expressions and validate every one.

    Nothing the model writes is trusted. Each expression is checked against the
    platform's operator list, the fields actually present in this scope, and the Power
    Pool limits — at most 8 operators and 3 distinct non-grouping fields. What comes
    back is the survivors and, just as usefully, why the others were rejected.
    """
    from ..templates.validate import (
        POWER_POOL_MAX_FIELDS,
        POWER_POOL_MAX_OPERATORS,
        count_operators,
        operators_in,
        unique_data_fields,
    )

    scope = body.scope.to_tuple()
    cached = await state.auth.cached_operators()
    known = {str(o.get("name")) for o in cached} if cached else set()

    seeds: list[str] = []
    if body.use_seeds:
        rows = await state.alphas.alphas(
            region=scope.region,
            delay=scope.delay,
            universe=scope.universe,
            instrument_type=scope.instrument_type,
            min_sharpe=1.0,
            limit=20,
        )
        seeds = [str(r["expression"]) for r in rows if r.get("expression")]

    result = await state.llm.power_pool_batch(
        scope,
        model_id=body.model,
        dataset_ids=body.dataset_ids,
        count=body.count,
        operators=sorted(known),
        brief=body.brief,
        seeds=seeds,
    )

    available = await state.studio.resolver.available(scope)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for item in result["proposed"]:
        expression = str(item["expression"]).strip()
        reasons: list[str] = []

        if known:
            unknown = sorted(set(operators_in(expression)) - known)
            if unknown:
                reasons.append(f"not operators on this platform: {', '.join(unknown)}")

        fields = unique_data_fields(expression)
        if available:
            missing = sorted(f for f in fields if f not in available)
            if missing:
                reasons.append(f"not in the catalog for {scope.label}: {', '.join(missing)}")

        operators = count_operators(expression)
        if operators > POWER_POOL_MAX_OPERATORS:
            reasons.append(f"{operators} operators, Power Pool allows {POWER_POOL_MAX_OPERATORS}")
        if len(fields) > POWER_POOL_MAX_FIELDS:
            reasons.append(f"{len(fields)} data fields, Power Pool allows {POWER_POOL_MAX_FIELDS}")

        entry = {
            "expression": expression,
            "idea": item.get("idea", ""),
            "operators": operators,
            "fields": sorted(fields),
        }
        (rejected if reasons else accepted).append(
            {**entry, "reasons": reasons} if reasons else entry
        )

    answer = result["answer"]
    return {
        "accepted": accepted,
        "rejected": rejected,
        "proposed": len(result["proposed"]),
        "scope": scope.label,
        "answer": answer.to_dict(),
        "checkedAgainst": {
            "operators": len(known),
            "fields": len(available),
            "maxOperators": POWER_POOL_MAX_OPERATORS,
            "maxFields": POWER_POOL_MAX_FIELDS,
        },
    }


class PowerPoolQueue(BaseModel):
    expressions: list[str]
    scope: Scope
    neutralization: str = "SUBINDUSTRY"
    decay: int = 0
    truncation: float = 0.08
    task: str = "power-pool"
    skip_duplicates: bool = True


@router.post("/power-pool/queue")
async def queue_power_pool(body: PowerPoolQueue, state: State) -> dict[str, Any]:
    """Queue the expressions you approved. A separate step, because this one costs."""
    from ..brain.schemas import SimulationRequest, SimulationSettings

    settings = SimulationSettings(
        instrumentType=body.scope.instrument_type,
        region=body.scope.region,
        delay=body.scope.delay,
        universe=body.scope.universe,
        neutralization=body.neutralization,
        decay=body.decay,
        truncation=body.truncation,
    )
    requests = [
        SimulationRequest(type="REGULAR", settings=settings, regular=e.strip())
        for e in body.expressions
        if e.strip()
    ]
    if not requests:
        return {"queued": [], "skipped": [], "message": "Nothing to queue."}

    result = await state.engine.enqueue(
        requests, task=body.task, skip_duplicates=body.skip_duplicates
    )
    return {**result, "task": body.task, "status": await state.engine.status()}


class ExplainRequest(BaseModel):
    subject: str = Field(description="A field, dataset, expression, or set of results")
    model: str | None = None


@router.post("/explain")
async def explain(body: ExplainRequest, state: State) -> dict[str, Any]:
    answer = await state.llm.explain(body.subject, model_id=body.model)
    return answer.to_dict()
