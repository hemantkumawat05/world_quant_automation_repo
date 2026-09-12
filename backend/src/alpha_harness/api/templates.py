"""The templates a study can be built on.

The starters are seeded on first run and read from here; ``/run`` is the only endpoint
that spends daily simulation quota. There is no authoring path — a template is a seed
for the Deepen lab, not a document the consultant writes (see ``INTENT.md`` §3).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..templates.library import TemplateNotFoundError, serialise
from .deps import OptionalUser, State, User

router = APIRouter(prefix="/api/templates", tags=["templates"])


class RunBody(BaseModel):
    """Expand a template and hand the result to the batch engine."""

    task: str | None = Field(
        default=None,
        description="Slot-quota group. Defaults to the template's own name.",
    )
    skip_duplicates: bool = True
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Queue at most this many, after expansion. Useful for a trial run.",
    )
    dry_run: bool = Field(
        default=False,
        description="Compute everything and queue nothing. Returns the same summary.",
    )


# --- listing and storage --------------------------------------------------


@router.get("")
async def list_templates(
    state: State,
    user: OptionalUser,
    tag: str | None = None,
    origin: str | None = None,
) -> list[dict[str, Any]]:
    user_id = user.user_id if user else None
    rows = await state.templates.list(tag=tag, origin=origin, user_id=user_id)
    return [serialise(r, include_source=False) for r in rows]


@router.post("/seed")
async def seed_templates(state: State, force: bool = False) -> dict[str, int]:
    """Install the starter templates.

    A no-op once the library has anything in it, so a restart never overwrites your own
    work. ``force=true`` reinstalls only the seeded ones.
    """
    return {"installed": await state.templates.seed(force=force)}


@router.get("/{template_id}")
async def get_template(template_id: int, state: State) -> dict[str, Any]:
    row = await state.templates.get(template_id)
    if row is None:
        raise TemplateNotFoundError(template_id)
    return serialise(row)


# --- running --------------------------------------------------------------


@router.post("/{template_id}/run")
async def run_template(template_id: int, body: RunBody, state: State) -> dict[str, Any]:
    """Expand a template and queue the result.

    The only endpoint here that costs anything. It refuses to queue a template with
    validation errors — an alpha the platform will reject still spends quota, and the
    daily cap is the binding constraint on a day's research.
    """
    spec = await state.templates.spec(template_id)
    review = await state.studio.review(spec)

    if not review.report.ok:
        raise HTTPException(
            422,
            detail={
                "code": "template_invalid",
                "message": "This template has problems that would waste simulations.",
                "problems": [p.to_dict() for p in review.report.problems],
            },
        )

    expansion = review.expansion
    if expansion is None or not expansion.requests:
        raise HTTPException(
            422,
            detail={
                "code": "template_empty",
                "message": (
                    "This template expands to nothing. Every combination was rejected by "
                    "its own constraints, or a variable resolved to no values."
                ),
                "rejected": len(expansion.rejected) if expansion else 0,
            },
        )

    requests = expansion.requests[: body.limit] if body.limit else expansion.requests
    summary = {
        "template": spec.name,
        "packing": expansion.packing_summary(),
        "rejected": len(expansion.rejected),
        "truncated": expansion.truncated,
        "totalPossible": expansion.total_possible,
        "queueing": len(requests),
        "warnings": [p.to_dict() for p in review.report.problems if p.severity == "warning"],
    }

    if body.dry_run:
        return {**summary, "dryRun": True, "queued": 0, "skipped": []}

    task = body.task or f"tpl:{spec.name}"[:64]
    result = await state.engine.enqueue(requests, task=task, skip_duplicates=body.skip_duplicates)
    return {
        **summary,
        "dryRun": False,
        "task": task,
        **result,
        "status": await state.engine.status(),
    }
