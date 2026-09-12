"""The alpha pool: browse, filter, tag, correlate, check.

**There is no submit endpoint, deliberately.** Submitting an alpha is irreversible, and
the safest guard against doing it by accident is that no route, service method or client
wrapper for it exists. Everything here reads, plus a single-alpha edit for naming,
tagging, colouring, favouriting and hiding — none of which changes what the alpha is.

Correlation is the one call to spend carefully: the platform throttles it per hour,
independently of the simulation quota, so it is never issued as part of a listing.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..brain.filters import OPERATOR_CHOICES, AlphaQuery, Filter, parse
from .deps import User

router = APIRouter(prefix="/api/alphas", tags=["alphas"])


class AlphaListRequest(BaseModel):
    """A page of the alpha pool.

    Filters are written the way the platform writes them — ``is.sharpe>=1.25``,
    ``status!=UNSUBMITTED``, ``name~momentum`` — so anything the platform supports works
    here without this application having to enumerate it.
    """

    filters: list[str] = Field(
        default_factory=list,
        description="Expressions like is.sharpe>=1.25, settings.region=USA, name~reversion",
    )
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    order: str = Field(default="-dateCreated", description="Prefix with - for descending")
    hidden: bool | None = Field(
        default=False,
        description=(
            "false excludes hidden alphas, true returns only those, null does not filter. "
            "The platform defaults to false, so it is always sent explicitly."
        ),
    )
    created_after: date | None = None
    created_before: date | None = Field(
        default=None, description="Inclusive — the named day is included in the range"
    )

    def to_query(self) -> AlphaQuery:
        parsed: list[Filter] = []
        for index, expression in enumerate(self.filters):
            try:
                parsed.append(parse(expression))
            except ValueError as exc:
                raise HTTPException(
                    422,
                    detail={
                        "code": "bad_filter",
                        "message": str(exc),
                        "index": index,
                        "filter": expression,
                    },
                ) from exc
        return AlphaQuery(
            limit=self.limit,
            offset=self.offset,
            order=self.order,
            filters=parsed,
            hidden=self.hidden,
            created_after=self.created_after,
            created_before=self.created_before,
        )


@router.post("/search")
async def search(body: AlphaListRequest, user: User) -> dict[str, Any]:
    """A filtered page of your alphas.

    A POST because the filter list is a body rather than a flat query string — the DSL
    puts the comparison operator inside the parameter name, which does not survive being
    round-tripped through an ordinary query-parameter parser.
    """
    query = body.to_query()
    page = await user.endpoints.list_alphas(query)
    return {**page, "query": query.query()}


@router.get("/summary")
async def summary(user: User) -> dict[str, Any]:
    """Counts by stage and status — the shape of your pool at a glance."""
    return await user.endpoints.alphas_summary()


@router.get("/filters")
async def filter_schema(user: User) -> dict[str, Any]:
    """Which fields can be filtered and ordered, from the platform itself.

    Read rather than hardcoded, for the same reason as the simulation settings: a stale
    copy silently drops filters the platform has since added.
    """
    schema = await user.endpoints.alpha_filter_schema()
    return {"schema": schema, "operators": list(OPERATOR_CHOICES)}


@router.get("/tags")
async def tags(user: User) -> list[dict[str, Any]]:
    return await user.endpoints.list_tags()


@router.get("/tags/{tag_id}/correlations")
async def tag_correlations(tag_id: str, user: User) -> dict[str, Any]:
    """How correlated the alphas in one tag are with each other.

    Rate limited by the platform per hour. Call it deliberately.
    """
    return await user.endpoints.tag_correlations(tag_id)


@router.get("/{alpha_id}")
async def get_alpha(alpha_id: str, user: User) -> dict[str, Any]:
    """Full alpha: settings, in-sample and out-of-sample statistics, and the checks."""
    alpha = await user.endpoints.get_alpha(alpha_id)
    return alpha.model_dump(by_alias=True)


@router.get("/{alpha_id}/recordsets")
async def list_recordsets(alpha_id: str, user: User) -> list[dict[str, Any]]:
    """Which time series this alpha has — pnl, sharpe, turnover, breakdowns."""
    refs = await user.endpoints.list_recordsets(alpha_id)
    return [r.model_dump(by_alias=True) for r in refs]


@router.get("/{alpha_id}/recordsets/{name}")
async def get_recordset(alpha_id: str, name: str, user: User) -> dict[str, Any]:
    """One time series, already zipped into rows.

    BRAIN returns these column-oriented (a schema plus positional arrays); the rows are
    assembled here so the frontend never has to. ``columnTypes`` carries the declared
    type per column — note ``permyriad`` means basis points, i.e. divided by 10,000.
    """
    recordset = await user.endpoints.get_recordset(alpha_id, name)
    return {
        "name": recordset.schema_.name or name,
        "title": recordset.schema_.title,
        "columns": [c.model_dump(by_alias=True) for c in recordset.schema_.properties],
        "columnTypes": recordset.column_types(),
        "rows": recordset.rows(),
    }


@router.get("/{alpha_id}/check")
async def check_alpha(alpha_id: str, user: User) -> dict[str, Any]:
    """Re-run the submission checks without submitting.

    The whole point of this endpoint: it tells you whether an alpha *would* pass, and
    changes nothing on the platform. It does update the local copy, so an alpha that has
    just resolved appears on the submit screen without waiting for the next backfill.
    """
    body = await user.endpoints.check_alpha(alpha_id)
    checks = ((body.get("is") or {}).get("checks")) or []
    if checks:
        await user.state.alphas.save_checks(alpha_id, checks)
    return body


@router.get("/{alpha_id}/correlations/{kind}")
async def correlations(alpha_id: str, kind: str, user: User) -> dict[str, Any]:
    """Correlation against your own submitted alphas (``self``) or production (``prod``).

    Throttled by the platform per hour, on its own budget separate from simulations, so
    this is never issued as part of a listing.
    """
    if kind not in ("self", "prod"):
        raise HTTPException(400, "kind must be 'self' or 'prod'")
    return await user.endpoints.correlations(alpha_id, kind)


@router.get("/{alpha_id}/similar")
async def similar(alpha_id: str, user: User, limit: int = Query(5, ge=1, le=50)) -> Any:
    """Alphas the platform considers related. Useful for spotting crowding."""
    return await user.endpoints.similar_alphas(alpha_id, limit)


class AlphaEdit(BaseModel):
    """Metadata only. Nothing here changes what the alpha computes.

    Single alpha at a time — the platform's bulk ``PATCH /alphas`` is deliberately not
    exposed, because a bulk write driven by a filter is easy to fire and impossible to
    undo.
    """

    name: str | None = None
    color: str | None = None
    tags: list[str] | None = None
    favorite: bool | None = None
    hidden: bool | None = None
    category: str | None = None

    def changes(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


@router.patch("/{alpha_id}")
async def edit_alpha(alpha_id: str, body: AlphaEdit, user: User) -> dict[str, Any]:
    changes = body.changes()
    if not changes:
        raise HTTPException(
            422,
            detail={"code": "nothing_to_change", "message": "No fields were given to change."},
        )
    alpha = await user.endpoints.patch_alpha(alpha_id, changes)
    return alpha.model_dump(by_alias=True)
