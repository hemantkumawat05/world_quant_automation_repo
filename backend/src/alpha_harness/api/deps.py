"""FastAPI dependencies and the exception -> HTTP mapping.

Every BRAIN failure becomes an HTTP response in exactly one place, so business logic can
raise typed exceptions and never touch ``HTTPException``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from ..brain.errors import (
    BrainAuthError,
    BrainDailyLimitReached,
    BrainError,
    BrainForbidden,
    BrainNotFound,
    BrainPollTimeout,
    BrainRateLimited,
    BrainServiceUnavailable,
    BrainTransportError,
    BrainValidationError,
    BrainVerificationRequired,
)
from ..catalog.queries import Tuple4
from ..engine.tracker import SubmissionFailed
from ..llm.keys import BudgetExhaustedError, LLMError, NoKeysError
from ..llm.service import UnknownPromptError
from ..optimize.errors import StudyError, StudyNotFoundError
from ..services.auth import NoCredentialError
from ..state import AppState
from ..templates.library import TemplateNotFoundError
from ..templates.schema import TemplateError


from ..brain.auth import SessionInfo
from ..brain.endpoints import BrainEndpoints
from ..security.jwt import verify_access_token


def get_state(request: Request) -> AppState:
    return request.app.state.harness


State = Annotated[AppState, Depends(get_state)]


class UserContext:
    """Carries the authenticated user identity and user-scoped endpoints."""

    def __init__(
        self,
        user_id: str,
        email: str,
        session: SessionInfo,
        endpoints: BrainEndpoints,
        state: AppState,
    ) -> None:
        self.user_id = user_id
        self.email = email
        self.session = session
        self.endpoints = endpoints
        self.state = state


async def get_current_user(request: Request, state: State) -> UserContext:
    auth_header = request.headers.get("Authorization")
    token: str | None = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.query_params.get("token")

    if not token:
        # If single local session exists, fallback to it
        if state.auth.session.authenticated and state.auth.session.user_id:
            user_id = state.auth.session.user_id
            email = await state.auth.stored_email(user_id=user_id) or ""
            return UserContext(
                user_id=user_id,
                email=email,
                session=state.auth.session,
                endpoints=state.endpoints,
                state=state,
            )
        raise BrainAuthError("Not authenticated")

    try:
        payload = verify_access_token(token, state.vault.key)
        user_id = str(payload["sub"])
        email = str(payload.get("email", ""))
        session_info, endpoints = await state.auth.get_user_context(user_id)
        if not session_info.authenticated:
            raise BrainAuthError("Session expired or invalid")
        return UserContext(
            user_id=user_id,
            email=email,
            session=session_info,
            endpoints=endpoints,
            state=state,
        )
    except Exception as exc:
        raise BrainAuthError("Invalid or expired authentication token") from exc


async def get_optional_user(request: Request, state: State) -> UserContext | None:
    try:
        return await get_current_user(request, state)
    except Exception:
        return None


User = Annotated[UserContext, Depends(get_current_user)]
OptionalUser = Annotated[UserContext | None, Depends(get_optional_user)]


class ScopedRequest(BaseModel):
    """A request aimed at one (instrument type, region, delay, universe).

    Almost every lab takes this same four-part address, and a field is not guaranteed to
    exist outside the one it was catalogued in — so the four travel together or not at
    all. Declared once here rather than copied into each router, which is how they drift.

    Accepts either spelling, the mirror of what :class:`..schemas.Out` does on the way
    out: the browser writes ``instrumentType`` and ``minSharpe``, Python reads
    ``instrument_type`` and ``min_sharpe``. Without this the client has to translate
    every field by hand and carry its own copy of each default, which is how a default
    ends up disagreeing with the ``Field(default=...)`` it was copied from.

    Deliberately no ``serialize_by_alias``: these are request models, and the snake_case
    attribute names are what the routers read.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    instrument_type: str = "EQUITY"
    region: str
    delay: int
    universe: str

    def scope(self) -> Tuple4:
        return Tuple4(
            instrument_type=self.instrument_type,
            region=self.region,
            delay=self.delay,
            universe=self.universe,
        )


def plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    """A lab plan with the requests replaced by their count.

    A preview says what *would* run; the requests themselves are megabytes of payload
    the screen has no use for. Shared because relocate and harden had identical copies.
    """
    return {k: v for k, v in plan.items() if k != "requests"} | {
        "simulations": len(plan["requests"])
    }


def _problem(
    status: int,
    code: str,
    message: str,
    **extra: Any,
) -> JSONResponse:
    """A consistent error envelope.

    ``code`` is stable and machine-readable; ``message`` is written for a human who may
    not know what a 429 is.
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, **extra}},
    )


#: ``exception -> (status, code, message, retryable)`` for the failures that need
#: nothing but those. Anything carrying a payload of its own — which fields were
#: rejected, how long to wait — stays an explicit handler below.
#:
#: ``message`` of ``None`` means the exception already phrases itself for a person; a
#: message written here belongs to a ``BrainError``, whose own text is the platform's
#: wording and goes in ``detail``. ``retryable`` of ``None`` omits the key, which is not
#: the same as ``False``: the client then falls back to "retryable if 5xx", right for a
#: poll timeout and wrong for a daily limit.
SIMPLE: dict[type[Exception], tuple[int, str, str | None, bool | None]] = {
    BrainAuthError: (
        401,
        "not_authenticated",
        "Not signed in to BRAIN, or the session expired. Sign in again.",
        None,
    ),
    BrainForbidden: (
        403,
        "forbidden",
        "Your BRAIN account is not permitted to do this. It may need a permission your "
        "tier does not have, or the settings combination is not available.",
        None,
    ),
    BrainNotFound: (404, "not_found", "BRAIN has no such resource.", None),
    # Deliberately distinct from an ordinary 429: retrying cannot help today.
    BrainDailyLimitReached: (
        429,
        "daily_limit_reached",
        "You have used your daily simulation quota. It resets at midnight US Eastern "
        "time — retrying before then will not work.",
        False,
    ),
    BrainServiceUnavailable: (
        503,
        "platform_unavailable",
        "The BRAIN simulation service is temporarily unavailable.",
        True,
    ),
    BrainTransportError: (
        502,
        "network_error",
        "Could not reach BRAIN. Check your connection.",
        True,
    ),
    BrainPollTimeout: (
        504,
        "poll_timeout",
        "BRAIN is still working on this. It may finish later — try again shortly.",
        None,
    ),
    NoCredentialError: (400, "no_credential", None, None),
    TemplateNotFoundError: (404, "template_not_found", None, None),
    StudyNotFoundError: (404, "study_not_found", None, None),
    # Understood and refused: the study as configured cannot produce anything.
    StudyError: (422, "study_invalid", None, None),
    NoKeysError: (400, "no_api_key", None, None),
    UnknownPromptError: (404, "prompt_not_found", None, None),
    LLMError: (400, "llm_error", None, None),
}


def _simple_handler(
    status: int, code: str, message: str | None, retryable: bool | None
) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handler(_r: Request, exc: Exception) -> JSONResponse:
        extra: dict[str, Any] = {}
        if retryable is not None:
            extra["retryable"] = retryable
        if message is not None and isinstance(exc, BrainError):
            extra["detail"] = exc.message
        return _problem(status, code, message or str(exc), **extra)

    return handler


def install_exception_handlers(app: FastAPI) -> None:
    # Starlette picks a handler by walking ``type(exc).__mro__``, so a specific handler
    # still wins over its base — BrainVerificationRequired over BrainAuthError,
    # BrainDailyLimitReached over BrainRateLimited — whatever order they register in.
    for exc_type, (status, code, message, retryable) in SIMPLE.items():
        app.add_exception_handler(exc_type, _simple_handler(status, code, message, retryable))

    @app.exception_handler(BrainVerificationRequired)
    async def _verification(_r: Request, exc: BrainVerificationRequired) -> JSONResponse:
        return _problem(
            409,
            "verification_required",
            "BRAIN needs to verify your identity before this session can be used. "
            "Open the link, complete the check, then sign in again.",
            verificationUrl=exc.url,
            inquiry=exc.inquiry,
        )

    @app.exception_handler(BrainValidationError)
    async def _validation(_r: Request, exc: BrainValidationError) -> JSONResponse:
        return _problem(
            422,
            "rejected_by_platform",
            "BRAIN rejected this request.",
            fields=exc.fields,
        )

    @app.exception_handler(BrainRateLimited)
    async def _rate_limited(_r: Request, exc: BrainRateLimited) -> JSONResponse:
        return _problem(
            429,
            "rate_limited",
            "BRAIN is throttling requests. Wait a moment and try again.",
            retryable=True,
            retryAfter=exc.retry_after,
        )

    @app.exception_handler(SubmissionFailed)
    async def _submission(_r: Request, exc: SubmissionFailed) -> JSONResponse:
        return _problem(
            422,
            "submission_failed",
            str(exc),
            recordId=exc.record_id,
            fields=exc.fields,
        )

    @app.exception_handler(BudgetExhaustedError)
    async def _budget(_r: Request, exc: BudgetExhaustedError) -> JSONResponse:
        # A daily exhaustion is not retryable in any useful sense; a per-minute one is.
        # Saying which, and for how long, is the difference between a useful error and a
        # spinner.
        return _problem(
            429,
            "llm_budget_exhausted",
            str(exc),
            retryable=not exc.daily,
            retryAfter=round(exc.retry_after),
            model=exc.model,
            keys=[s.to_dict() for s in exc.states],
        )

    @app.exception_handler(TemplateError)
    async def _bad_template(_r: Request, exc: TemplateError) -> JSONResponse:
        # A malformed template is ordinary input, not a server fault. Typed rather than
        # catching ValueError, which would quietly turn real bugs into 422s.
        return _problem(422, "template_invalid", str(exc), problems=exc.problems)

    @app.exception_handler(BrainError)
    async def _generic(_r: Request, exc: BrainError) -> JSONResponse:
        return _problem(
            502,
            "platform_error",
            "BRAIN returned an unexpected response.",
            detail=exc.message,
            platformStatus=exc.status,
        )
