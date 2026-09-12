"""Today: what you have, and what you are about to waste.

One endpoint behind the first screen a consultant sees. It answers three questions in
the order they matter:

1. **How many simulations do you have left today?** They do not carry over. An unused
   allowance is compute that WorldQuant paid for and nobody used, and that is the whole
   reason this application exists — so it is the largest number on the page.
2. **What can you do?** The account's permissions, in words rather than codes.
3. **What is left in the assistant's budget?** Also daily, also wasted if unused.

It also drives the onboarding flow. Everything here is linear on purpose: sign in, add a
key, start. A branch is a decision, and every decision is somewhere to give up.

**On the quota being an estimate.** The platform reveals the true daily limit *only* in
the response headers of a simulation POST — verified against ``OPTIONS /simulations``,
``/users/self``, ``/configuration`` and the alpha summary, none of which carry it. So
the figure starts as the configured allowance minus what this application has run today,
and is replaced by the platform's own number the moment the first batch comes back.
``exact`` says which one you are looking at, because a number presented as measured when
it was assumed is worse than no number.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query

from ..brain.auth import SessionInfo
from ..brain.filters import PLATFORM_TZ
from ..db.models import SyncStatus
from ..llm.budget import seconds_until_reset
from .deps import OptionalUser, State, User

router = APIRouter(prefix="/api/today", tags=["today"])


#: Account permissions in plain language. A consultant should not have to look up
#: ``MULTI_SIMULATION`` to find out they can run ten at a time.
FEATURES: dict[str, tuple[str, str]] = {
    "CONSULTANT": (
        "Consultant",
        "You are a paid consultant. Your alphas can earn you money when they are used.",
    ),
    "MULTI_SIMULATION": (
        "Ten at a time",
        "You can run ten simulations in one go instead of one, which is why a day's work "
        "takes hours instead of weeks.",
    ),
    "SUPER_ALPHA": (
        "SuperAlpha",
        "You can combine several of your alphas into one bigger one.",
    ),
    "PROD_ALPHAS": (
        "Production alphas",
        "You can see the alphas that are running with real money.",
    ),
    "VISUALIZATION": ("Charts", "You can see how an alpha performed, drawn as a chart."),
    "REGION_AGNOSTIC": (
        "Every market",
        "You can build alphas for any country, not just one. This is the single biggest "
        "source of ideas nobody else is having.",
    ),
    "BRAIN_LABS": ("BRAIN Labs", "You have access to the experimental tools."),
    "BRAIN_LABS_JUPYTER_LAB": ("Notebooks", "You can write Python against the platform."),
    "BEFORE_AND_AFTER_PERFORMANCE_V2": (
        "Before and after",
        "You can see how an alpha behaved before and after it was submitted.",
    ),
    "REFERRAL": ("Referrals", "You can invite other people to the platform."),
    "WORKDAY": ("Workday", "Your account is linked to WorldQuant's payroll system."),
}


@router.get("")
async def today(
    state: State,
    user: OptionalUser,
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    instrument_type: str = "EQUITY",
) -> dict[str, Any]:
    """Everything the first screen needs, in one call."""
    try:
        if user and user.session.authenticated:
            session = user.session
            stored_email = user.email or await state.auth.stored_email(user_id=user.user_id)
            keys = await state.llm.keys.list(user_id=user.user_id)
            enabled_keys = [k for k in keys if k.enabled]
            step = "ready" if enabled_keys else "add-key"
            user_id = user.user_id
        else:
            session = SessionInfo.anonymous()
            stored_email = None
            keys = []
            enabled_keys = []
            step = "sign-in"
            user_id = None

        return {
            "step": step,
            "you": await _you(state, session, stored_email, user_id=user_id),
            "simulations": await _simulations(state, user_id=user_id),
            "assistant": await _assistant(state, keys, enabled_keys),
            "catalog": await _catalog(state, instrument_type, region, delay, universe),
        }
    except Exception as exc:
        import structlog
        structlog.get_logger("alpha_harness").exception("today.unhandled_error", error=str(exc))
        return {
            "step": "sign-in",
            "you": {
                "signedIn": False,
                "email": None,
                "userId": None,
                "fullName": None,
                "features": [],
                "canRunTenAtOnce": False,
                "verificationUrl": None,
            },
            "simulations": {
                "limit": 5000,
                "used": 0,
                "remaining": 5000,
                "pendingCharge": 0,
                "queued": 0,
                "unspoken": 5000,
                "exact": False,
                "resetsInSeconds": 86400,
                "resetsAt": "midnight US Eastern",
                "engine": {
                    "slots": 8,
                    "maxBatch": 10,
                    "slotsUsed": 0,
                    "slotsFree": 8,
                    "queued": {},
                    "queuedTotal": 0,
                    "inFlight": {},
                    "quotas": {},
                    "dailyLimitHit": False,
                },
                "headline": "Sign in to start research.",
            },
            "assistant": {
                "keys": 0,
                "enabledKeys": 0,
                "requestsRemainingToday": 0,
                "budget": [],
                "resetsInSeconds": 86400,
                "resetsAt": "midnight Pacific",
                "headline": "No assistant key yet.",
            },
            "catalog": {
                "scope": f"{instrument_type}/{region}/D{delay}/{universe}",
                "synced": False,
                "fields": 0,
                "running": None,
                "anySynced": False,
            },
        }


@router.get("/bar")
async def bar(state: State, user: OptionalUser) -> dict[str, Any]:
    """The top bar: session time left, simulations left, and the reset countdown."""
    if user and user.session.authenticated:
        session = user.session
        sims = await _simulations(state, user_id=user.user_id)
        return {
            "signedIn": True,
            "fullName": session.full_name or user.email or session.user_id,
            "expiresInSeconds": session.expires_in_seconds,
            "simulations": {k: sims[k] for k in ("remaining", "limit", "exact", "queued")},
            "resetsInSeconds": sims["resetsInSeconds"],
        }
    sims = await _simulations(state, user_id=None)
    return {
        "signedIn": False,
        "fullName": None,
        "expiresInSeconds": None,
        "simulations": {k: sims[k] for k in ("remaining", "limit", "exact", "queued")},
        "resetsInSeconds": sims["resetsInSeconds"],
    }


@router.get("/activity")
async def activity(
    state: State,
    user: User,
    days: int = Query(default=90, ge=7, le=365),
) -> dict[str, Any]:
    """Simulations sent and alphas submitted per day, from BRAIN's own activity record."""
    end = datetime.now(PLATFORM_TZ).date()
    start = end - timedelta(days=days - 1)
    simulations = await user.endpoints.activity_counts("simulations", start.isoformat())
    submissions = await user.endpoints.activity_counts("submissions", start.isoformat())
    return {"days": daily_activity(simulations, submissions, start, end)}


def daily_activity(
    simulations: list[tuple[str, int]],
    submissions: list[tuple[str, int]],
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """One row per day from ``start`` to ``end``, zero where BRAIN listed nothing."""
    sims, subs = dict(simulations), dict(submissions)
    days = (start + timedelta(days=n) for n in range((end - start).days + 1))
    return [
        {
            "date": d.isoformat(),
            "simulations": sims.get(d.isoformat(), 0),
            "submissions": subs.get(d.isoformat(), 0),
        }
        for d in days
    ]


async def _catalog(
    state: State, instrument_type: str, region: str, delay: int, universe: str
) -> dict[str, Any]:
    """Whether this market has been downloaded, and how far along if it is downloading.

    Every lab that spends the allowance needs a synced scope, so a consultant who has
    not downloaded one has nothing to press. This is what lets the first screen say so
    and offer the download, rather than showing a disabled button with no explanation.
    """
    from ..catalog.sync import serialise_run

    try:
        synced = await state.queries.synced_tuples()
    except Exception:
        # A catalog that will not open is diagnosed on the data screen; here it simply
        # means nothing is downloaded yet.
        synced = []

    match = next(
        (
            row
            for row in synced
            if row["region"] == region
            and int(row["delay"]) == delay
            and row["universe"] == universe
            and row["instrument_type"] == instrument_type
        ),
        None,
    )

    try:
        running = next(
            (r for r in await state.sync.runs(limit=10) if r.status == SyncStatus.RUNNING), None
        )
    except Exception:
        running = None

    return {
        "scope": f"{instrument_type}/{region}/D{delay}/{universe}",
        "synced": match is not None,
        "fields": int(match["fields"]) if match else 0,
        "running": serialise_run(running) if running else None,
        #: Some other market is downloaded, so the labs are not blocked outright — the
        #: consultant just picked a scope they have not fetched yet.
        "anySynced": bool(synced),
    }


async def _you(
    state: State,
    session: Any,
    stored_email: str | None,
    user_id: str | None = None,
) -> dict[str, Any]:
    granted = list(session.permissions or [])
    full_name = session.full_name
    if not full_name and session.authenticated and user_id:
        await state.auth.get_user_profile(user_id)
        full_name = session.full_name
    if not full_name and stored_email:
        full_name = stored_email.split("@")[0].replace(".", " ").title()

    return {
        "signedIn": session.authenticated,
        "email": stored_email,
        "userId": session.user_id,
        "fullName": full_name or session.user_id,
        "features": [
            {
                "code": code,
                "label": FEATURES.get(code, (code.replace("_", " ").title(), ""))[0],
                "meaning": FEATURES.get(code, (code, "An extra permission on your account."))[1],
            }
            for code in granted
        ],
        "canRunTenAtOnce": "MULTI_SIMULATION" in granted,
        "verificationUrl": session.verification_url,
    }


async def _simulations(state: State, user_id: str | None = None) -> dict[str, Any]:
    allowance = state.settings.daily_simulation_allowance
    try:
        used = await state.tracker.used_today(user_id=user_id)
        snapshot = await state.tracker.latest_quota()
    except Exception:
        used = 0
        snapshot = None

    exact = False
    pending = 0
    limit, remaining = allowance, max(0, allowance - used)
    if snapshot is not None and snapshot.remaining is not None:
        observed = snapshot.observed_at
        if observed and _same_platform_day(observed):
            exact = True
            limit = snapshot.limit_total or allowance
            try:
                pending = await state.tracker.uncharged_since(observed, user_id=user_id)
            except Exception:
                pending = 0
            remaining = max(0, snapshot.remaining - pending)

    resets_in = seconds_until_reset(tz=PLATFORM_TZ)
    try:
        engine = await state.engine.status()
    except Exception:
        engine = {
            "slots": 8,
            "maxBatch": 10,
            "slotsUsed": 0,
            "slotsFree": 8,
            "queued": {},
            "queuedTotal": 0,
            "inFlight": {},
            "quotas": {},
            "dailyLimitHit": False,
        }
    # Work that is queued has not been sent yet, so it does not show up in ``used`` —
    # but it *is* spoken for, and counting it as waste would tell someone who has just
    # queued their whole day that they have done nothing. That is the opposite of what
    # this page is for.
    queued = int(engine.get("queuedTotal") or 0)
    unspoken = max(0, remaining - queued)

    return {
        "limit": limit,
        "used": used if not exact else max(0, limit - remaining),
        "remaining": remaining,
        #: Sent but not yet counted by BRAIN's lagging header; already taken off ``remaining``.
        "pendingCharge": pending,
        "queued": queued,
        #: What is genuinely still going to waste: not yet run and not yet claimed.
        "unspoken": unspoken,
        # False means "assumed from your allowance", true means "the platform told us".
        "exact": exact,
        "resetsInSeconds": round(resets_in),
        "resetsAt": "midnight US Eastern",
        "engine": engine,
        "headline": _headline(remaining, limit, queued),
    }


def _headline(remaining: int, limit: int, queued: int = 0) -> str:
    """The sentence at the top of the Dashboard.

    Deterministic: it depends only on the allowance, what is left, and what is queued. It
    names what is still unused while there is some, and stops nagging once the day is
    claimed. No clock in it — the reset countdown is shown on its own.
    """
    if limit <= 0:
        return "No simulations available today."
    if remaining <= 0:
        return "You have used every simulation today. Nothing was wasted."

    unspoken = max(0, remaining - queued)
    if unspoken <= 0:
        return "Every simulation left today is queued. Nothing is going to waste."

    share = unspoken / limit
    tail = f" {queued:,} more are already queued." if queued else ""
    if share > 0.9:
        return (
            f"{unspoken:,} simulations are unused today. They cannot be saved for tomorrow.{tail}"
        )
    if share > 0.5:
        return f"{unspoken:,} simulations left today.{tail}"
    return f"{unspoken:,} left of {limit:,}. Good day so far.{tail}"


def _same_platform_day(moment: datetime) -> bool:
    from datetime import UTC

    now = datetime.now(UTC).astimezone(PLATFORM_TZ)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(PLATFORM_TZ).date() == now.date()


async def _assistant(state: State, keys: list[Any], enabled: list[Any]) -> dict[str, Any]:
    """The Google AI Studio side: also a daily budget, also wasted if unused.

    Reported per key as well as in total, because the whole reason to add a second
    account is that the allowance is per account and simply doubles.
    """
    budget = []
    remaining_total = 0
    if enabled:
        status = await state.llm.keys.status(state.llm.registry)
        budget = status["budget"]
        # The models with room to work in are the ones worth totalling; a
        # twenty-a-day model is not a budget anyone plans around.
        remaining_total = sum(b["remainingToday"] for b in budget if b["bulk"])

    return {
        "keys": len(keys),
        "enabledKeys": len(enabled),
        "requestsRemainingToday": remaining_total,
        "budget": budget,
        "resetsInSeconds": round(seconds_until_reset()),
        "resetsAt": "midnight Pacific",
        "headline": _assistant_headline(len(enabled), remaining_total),
    }


def _assistant_headline(enabled: int, remaining: int) -> str:
    if enabled == 0:
        return (
            "No assistant key yet. It is free, takes a minute, and it is what explains "
            "the data to you in plain English."
        )
    if remaining <= 0:
        return "The assistant has used its free requests for today. It resets at midnight Pacific."
    plural = "" if enabled == 1 else "s"
    return f"{remaining:,} free assistant requests left today across {enabled} key{plural}."
