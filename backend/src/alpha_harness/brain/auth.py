"""Sign-in and session lifecycle.

Sessions persist on the platform, so the expensive part — solving the ALTCHA
proof-of-work — should happen rarely. The flow is always:

    restore the cookie jar -> GET /authentication -> still valid? done.
                                                  -> expired? solve captcha, sign in.

We always solve the captcha before posting credentials rather than probing without one
first. A rejected sign-in counts against the lockout budget and returns ``429`` from the
same endpoint that reports bad passwords, so a speculative attempt is a bad trade for
about a second of hashing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from ..schemas import camel_dict
from .endpoints import BrainEndpoints
from .errors import BrainAuthError, BrainVerificationRequired
from .schemas import AuthState

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class SessionInfo:
    """What the UI needs to render the login tab."""

    authenticated: bool
    user_id: str | None = None
    full_name: str | None = None
    permissions: list[str] = field(default_factory=list)
    expires_at: float | None = None
    restored_from_cache: bool = False
    verification_url: str | None = None
    detail: str | None = None

    @property
    def can_multi_simulate(self) -> bool:
        return "MULTI_SIMULATION" in self.permissions

    @property
    def is_consultant(self) -> bool:
        return "CONSULTANT" in self.permissions

    @property
    def expires_in_seconds(self) -> int | None:
        if self.expires_at is None:
            return None
        return max(0, round(self.expires_at - time.time()))

    def to_dict(self) -> dict[str, Any]:
        return camel_dict(self) | {
            "canMultiSimulate": self.can_multi_simulate,
            "isConsultant": self.is_consultant,
            "expiresInSeconds": self.expires_in_seconds,
        }

    @classmethod
    def anonymous(cls, detail: str | None = None) -> SessionInfo:
        return cls(authenticated=False, detail=detail)

    @classmethod
    def from_state(cls, state: AuthState, *, restored: bool = False) -> SessionInfo:
        expires_at = None
        if state.token and state.token.expiry:
            expires_at = time.time() + float(state.token.expiry)
        return cls(
            authenticated=state.user_id is not None,
            user_id=state.user_id,
            permissions=list(state.permissions),
            expires_at=expires_at,
            restored_from_cache=restored,
        )


class Authenticator:
    """Establishes and validates a BRAIN session on one :class:`BrainEndpoints`."""

    def __init__(self, endpoints: BrainEndpoints) -> None:
        self.endpoints = endpoints

    async def restore(self, cookies: list[dict[str, Any]] | None) -> SessionInfo | None:
        """Try to reuse a cached cookie jar.

        Returns ``None`` when there is nothing to restore or the session has lapsed, so
        the caller can fall through to a full sign-in.
        """
        if not cookies:
            return None

        self.endpoints.client.load_cookies(cookies)
        try:
            state = await self.endpoints.get_auth()
        except BrainVerificationRequired as exc:
            # The session exists but needs a browser check. Keep the cookies: clearing
            # them here forced a full sign-in after the user had already verified.
            return _needs_verification(exc)
        if state is None or state.user_id is None:
            log.info("brain.session.expired")
            self.endpoints.client.clear_cookies()
            return None

        log.info("brain.session.restored", user_id=state.user_id)
        return SessionInfo.from_state(state, restored=True)

    async def login(self, email: str, password: str) -> SessionInfo:
        """Full sign-in: solve the proof-of-work, then exchange Basic auth for a cookie."""
        started = time.monotonic()
        solution = await self.endpoints.solve_captcha()
        log.info(
            "brain.captcha.solved",
            number=solution.number,
            took_ms=solution.took_ms,
        )

        try:
            state = await self.endpoints.authenticate(email, password, captcha=solution.encode())
        except BrainVerificationRequired as exc:
            # Not a credential failure. The user must finish this in a browser.
            log.warning("brain.auth.verification_required", inquiry=exc.inquiry)
            return SessionInfo(
                authenticated=False,
                verification_url=exc.url,
                detail=(
                    "BRAIN requires identity verification. Open the link, complete the "
                    "check, then sign in again."
                ),
            )
        except BrainAuthError as exc:
            log.warning("brain.auth.failed", detail=exc.message)
            return SessionInfo.anonymous(detail="Incorrect email or password.")

        if state.user_id is None:
            return SessionInfo.anonymous(detail="BRAIN accepted the request but returned no user.")

        log.info(
            "brain.auth.ok",
            user_id=state.user_id,
            permissions=state.permissions,
            elapsed_s=round(time.monotonic() - started, 2),
        )
        return SessionInfo.from_state(state)

    async def verify(self, url: str, email: str, password: str) -> SessionInfo | None:
        """Finish a pending identity verification on the same inquiry.

        Signing in again opens a new inquiry, so the one the user just completed is never
        used. BRAIN answers ``201`` to a POST of the Persona link once it is done. Returns
        ``None`` when the inquiry is gone, so the caller can fall back to a full sign-in.
        """
        r = await self.endpoints.client.request(
            "POST", url, auth=(email, password), raise_for_status=False
        )
        if r.status in (200, 201):
            state = await self.endpoints.get_auth()
            if state is not None and state.user_id is not None:
                log.info("brain.auth.verified", user_id=state.user_id)
                return SessionInfo.from_state(state)
            if isinstance(r.body, dict) and (r.body.get("user") or r.body.get("token")):
                return SessionInfo.from_state(AuthState.model_validate(r.body))
            return None
        error = self.endpoints.client._to_error("POST", url, r) if r.status >= 400 else None
        if isinstance(error, BrainVerificationRequired):
            log.info("brain.auth.verification_pending", inquiry=error.inquiry)
            return _needs_verification(error) if error.inquiry else _pending(url)
        log.warning("brain.auth.verification_lost", status=r.status)
        return None

    async def ensure(
        self, email: str, password: str, *, cookies: list[dict[str, Any]] | None = None
    ) -> SessionInfo:
        """Restore if possible, otherwise sign in."""
        restored = await self.restore(cookies)
        if restored is not None:
            return restored
        return await self.login(email, password)

    async def status(self) -> SessionInfo:
        """Current session state, without attempting to sign in."""
        try:
            state = await self.endpoints.get_auth()
        except BrainVerificationRequired as exc:
            return _needs_verification(exc)
        if state is None or state.user_id is None:
            return SessionInfo.anonymous()
        return SessionInfo.from_state(state)

    async def logout(self) -> None:
        await self.endpoints.logout()
        log.info("brain.session.ended")


def _needs_verification(exc: BrainVerificationRequired) -> SessionInfo:
    return _pending(exc.url)


def _pending(url: str) -> SessionInfo:
    return SessionInfo(
        authenticated=False,
        verification_url=url,
        detail=(
            "BRAIN requires identity verification. Open the link, complete the check, "
            "then sign in again."
        ),
    )
