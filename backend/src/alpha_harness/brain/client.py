"""Async HTTP client for the BRAIN API.

Two behaviours in here are the reason this file exists, and both break naive clients:

1. **``200 OK`` with a ``Retry-After`` header means "not ready yet".** Not ``202``. The
   presence of the header — not the status code — is the signal, and the body is empty
   until the job finishes. :meth:`BrainClient.poll` is the only correct way to read
   recordsets, correlations, checks, and simulations.
2. **Submission flips verb mid-flight.** ``POST /alphas/{id}/submit`` starts it; every
   poll afterwards is a ``GET`` to the same path. Polling with ``POST`` would re-trigger
   the submission. :meth:`poll` takes a separate ``poll_method`` for this.

Versioning lives in the ``Accept`` header (``application/json;version=N``), not the path.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Self
from urllib.parse import urljoin

import httpx
import structlog

from .errors import (
    DAILY_LIMIT_DETAIL,
    BrainAuthError,
    BrainDailyLimitReached,
    BrainError,
    BrainForbidden,
    BrainNotFound,
    BrainPollTimeout,
    BrainRateLimited,
    BrainServerError,
    BrainServiceUnavailable,
    BrainTransportError,
    BrainValidationError,
    BrainVerificationRequired,
)

log = structlog.get_logger(__name__)

Method = Literal["GET", "POST", "PATCH", "DELETE", "OPTIONS", "PUT"]

DEFAULT_VERSION = "2.0"


class Throttle:
    """Paces outbound requests and absorbs backoff penalties.

    The catalog crawl needs ~200 requests per scope (``/data-fields`` caps ``limit`` at
    50 and USA/D1/TOP3000 alone holds 10,000 fields). Fired back-to-back that earns a
    ``429`` within seconds, and the platform's own guidance is to keep request rates near
    what the web UI would generate. So requests are spaced, and a ``429`` adds a penalty
    that every subsequent caller waits out.
    """

    def __init__(self, min_interval: float = 0.0) -> None:
        self.min_interval = min_interval
        self._last = float("-inf")
        self._penalty_until = float("-inf")
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.min_interval <= 0 and self._penalty_until == float("-inf"):
            return
        # Reserve a slot under the lock, then sleep outside it. Sleeping while holding the
        # lock paced identically but made every caller queue behind one sleeper and left
        # nothing cancellable until the head of the queue woke.
        async with self._lock:
            ready_at = max(time.monotonic(), self._last + self.min_interval, self._penalty_until)
            self._last = ready_at
        wait = ready_at - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)

    def penalise(self, seconds: float) -> None:
        """Hold every caller back for ``seconds`` after a throttling response."""
        self._penalty_until = max(self._penalty_until, time.monotonic() + seconds)


@dataclass(frozen=True, slots=True)
class RateLimit:
    """Daily simulation quota, as reported on ``POST /simulations``."""

    limit: int | None
    remaining: int | None
    reset_seconds: float | None
    observed_at: float

    @classmethod
    def from_headers(cls, headers: httpx.Headers) -> RateLimit | None:
        def _int(name: str) -> int | None:
            raw = headers.get(name)
            try:
                return int(raw) if raw is not None else None
            except ValueError:
                return None

        limit = _int("x-ratelimit-limit")
        remaining = _int("x-ratelimit-remaining")
        reset_raw = headers.get("x-ratelimit-reset")
        try:
            reset = float(reset_raw) if reset_raw is not None else None
        except ValueError:
            reset = None

        if limit is None and remaining is None and reset is None:
            return None
        return cls(limit=limit, remaining=remaining, reset_seconds=reset, observed_at=time.time())


@dataclass(slots=True)
class BrainResponse:
    """A completed BRAIN response with the bits callers actually need."""

    status: int
    headers: httpx.Headers
    body: Any
    retry_after: float | None
    rate_limit: RateLimit | None
    location: str | None

    @property
    def pending(self) -> bool:
        """True while the server is still working on an asynchronous job."""
        return self.retry_after is not None


def _parse_retry_after(headers: httpx.Headers, floor: float) -> float | None:
    """Return the retry delay in seconds, or ``None`` if the result is ready.

    BRAIN sends integer seconds. A malformed or zero value is treated as "ready" rather
    than looping forever on a header we cannot interpret.
    """
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        log.warning("brain.retry_after.unparseable", value=raw)
        return None
    if seconds <= 0:
        return None
    return max(seconds, floor)


class BrainClient:
    """Cookie-authenticated async client.

    The instance owns a cookie jar; :meth:`export_cookies` / :meth:`load_cookies` persist
    it so a restart does not cost another proof-of-work solve.
    """

    def __init__(
        self,
        base_url: str = "https://api.worldquantbrain.com",
        *,
        timeout: float = 30.0,
        min_retry_after: float = 1.0,
        poll_timeout: float = 300.0,
        min_request_interval: float = 0.0,
        default_attempts: int = 5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self._min_retry_after = min_retry_after
        self.throttle = Throttle(min_request_interval)
        self._default_attempts = max(1, default_attempts)
        self._poll_timeout = poll_timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=False,
            transport=transport,
            headers={"User-Agent": "alpha-harness/0.1 (local research studio)"},
        )
        self.last_rate_limit: RateLimit | None = None

    @property
    def min_retry_after(self) -> float:
        return self._min_retry_after

    @property
    def poll_timeout(self) -> float:
        return self._poll_timeout

    @property
    def default_attempts(self) -> int:
        return self._default_attempts

    # -- lifecycle -------------------------------------------------------

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- cookie jar ------------------------------------------------------

    def export_cookies(self) -> list[dict[str, Any]]:
        """Serialise the jar for storage."""
        return [
            {
                "name": c.name,
                "value": c.value,
                "domain": c.domain,
                "path": c.path,
                "expires": c.expires,
                "secure": c.secure,
            }
            for c in self._client.cookies.jar
        ]

    def load_cookies(self, cookies: list[dict[str, Any]]) -> None:
        """Restore a jar produced by :meth:`export_cookies`."""
        for c in cookies:
            if not c.get("name") or c.get("value") is None:
                continue
            self._client.cookies.set(
                c["name"], c["value"], domain=c.get("domain", ""), path=c.get("path", "/")
            )

    def clear_cookies(self) -> None:
        self._client.cookies.clear()

    @property
    def has_session_cookie(self) -> bool:
        return len(self._client.cookies.jar) > 0

    # -- core request ----------------------------------------------------

    async def request(
        self,
        method: Method,
        path: str,
        *,
        version: str = DEFAULT_VERSION,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        raise_for_status: bool = True,
    ) -> BrainResponse:
        """Issue one request and translate failures into typed exceptions.

        ``params`` values that are ``None`` are dropped. Sequences are repeated, which is
        what the alpha filter DSL expects.
        """
        merged = {"Accept": f"application/json;version={version}"}
        if headers:
            merged.update(headers)

        clean_params = {k: v for k, v in params.items() if v is not None} if params else None

        await self.throttle.acquire()

        try:
            response = await self._client.request(
                method,
                path.lstrip("/"),
                params=clean_params,
                json=json_body,
                headers=merged,
                auth=auth or httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.TimeoutException as exc:
            raise BrainTransportError(f"{method} {path} timed out") from exc
        except httpx.HTTPError as exc:
            raise BrainTransportError(f"{method} {path} failed: {exc}") from exc

        rate_limit = RateLimit.from_headers(response.headers)
        if rate_limit is not None:
            self.last_rate_limit = rate_limit

        body = _decode(response)
        result = BrainResponse(
            status=response.status_code,
            headers=response.headers,
            body=body,
            retry_after=_parse_retry_after(response.headers, self._min_retry_after),
            rate_limit=rate_limit,
            location=response.headers.get("location"),
        )

        if raise_for_status and response.status_code >= 400:
            raise self._to_error(method, path, result)
        return result

    def _to_error(self, method: str, path: str, r: BrainResponse) -> BrainError:
        where = f"{method} {path}"
        body = r.body
        detail = body.get("detail") if isinstance(body, dict) else None

        if r.status == 401:
            inquiry = body.get("inquiry") if isinstance(body, dict) else None
            location = r.headers.get("location")
            if inquiry or (r.headers.get("www-authenticate", "").lower() == "persona"):
                if not inquiry and location and "inquiry=" in location:
                    inquiry = location.split("inquiry=", 1)[1].split("&", 1)[0]
                relative = location or f"authentication/persona?inquiry={inquiry}"
                url = urljoin(self.base_url, relative)
                return BrainVerificationRequired(
                    "BRAIN requires identity verification before this session can be used. "
                    "Complete it in a browser, then sign in again.",
                    inquiry=str(inquiry or ""),
                    url=url,
                    body=body,
                )
            return BrainAuthError(
                f"{where}: not authenticated ({detail or 'session expired or bad credentials'})",
                status=401,
                body=body,
            )

        if r.status == 403:
            return BrainForbidden(
                f"{where}: forbidden ({detail or 'account lacks permission for this request'})",
                status=403,
                body=body,
            )

        if r.status == 404:
            return BrainNotFound(f"{where}: not found", status=404, body=body)

        if r.status == 400:
            fields = body if isinstance(body, dict) else {"detail": body}
            return BrainValidationError(
                f"{where}: rejected by the platform", fields=fields, body=body
            )

        if r.status == 429:
            if detail == DAILY_LIMIT_DETAIL:
                return BrainDailyLimitReached(
                    "Daily simulation limit reached. The quota resets at US-Eastern "
                    "midnight; retrying before then cannot succeed.",
                    body=body,
                )
            return BrainRateLimited(
                f"{where}: rate limited ({detail or 'too many requests'})",
                retry_after=_parse_retry_after(r.headers, self._min_retry_after),
                body=body,
            )

        if r.status == 503:
            return BrainServiceUnavailable(f"{where}: service unavailable", status=503, body=body)

        if r.status >= 500:
            return BrainServerError(f"{where}: server error {r.status}", status=r.status, body=body)

        return BrainError(f"{where}: unexpected status {r.status}", status=r.status, body=body)

    async def request_retrying(
        self,
        method: Method,
        path: str,
        *,
        attempts: int | None = None,
        base_backoff: float = 2.0,
        max_backoff: float = 60.0,
        **kwargs: Any,
    ) -> BrainResponse:
        """Issue a request, retrying the failures that are worth retrying.

        Retries ``429`` (ordinary throttling), ``503`` and transport errors with
        exponential backoff plus jitter, preferring the server's own ``Retry-After``
        when it supplies one. A ``429`` also penalises the shared throttle, so
        concurrent callers slow down together rather than each discovering the limit.

        Never retries :class:`BrainDailyLimitReached` — the quota resets on US-Eastern
        midnight and no amount of waiting within a run will help.
        """
        attempts = attempts or self._default_attempts
        last: BrainError | None = None

        for attempt in range(1, attempts + 1):
            try:
                return await self.request(method, path, **kwargs)
            except BrainDailyLimitReached:
                raise
            except BrainError as exc:
                if not exc.retryable or attempt == attempts:
                    raise
                last = exc

                delay = getattr(exc, "retry_after", None)
                if delay is None:
                    delay = min(base_backoff * (2 ** (attempt - 1)), max_backoff)
                delay = min(float(delay), max_backoff)
                # Jitter so parallel callers do not resynchronise on the same instant.
                delay += random.uniform(0, delay * 0.25)

                if isinstance(exc, BrainRateLimited):
                    self.throttle.penalise(delay)

                log.warning(
                    "brain.retrying",
                    path=path,
                    attempt=attempt,
                    of=attempts,
                    delay=round(delay, 2),
                    reason=type(exc).__name__,
                )
                await asyncio.sleep(delay)

        assert last is not None
        raise last

    # -- the asynchronous job protocol -----------------------------------

    async def poll(
        self,
        method: Method,
        path: str,
        *,
        version: str = DEFAULT_VERSION,
        poll_method: Method | None = None,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        timeout: float | None = None,
        on_progress: Any = None,
    ) -> BrainResponse:
        """Drive an asynchronous job to completion.

        Re-issues the request while the response carries ``Retry-After``, honouring the
        server's requested delay. Returns the first response *without* that header —
        that is the one with a real body.

        ``poll_method`` overrides the verb used for follow-up requests. Pass ``"GET"``
        alongside ``method="POST"`` for ``/alphas/{id}/submit``; the body is sent only on
        the initial request.
        """
        deadline = time.monotonic() + (timeout if timeout is not None else self._poll_timeout)
        follow_up = poll_method or method
        attempt = 0

        while True:
            first = attempt == 0
            response = await self.request(
                method if first else follow_up,
                path,
                version=version,
                params=params,
                json_body=json_body if first else None,
                # A 201 from POST /simulations carries no Retry-After and no body; that
                # is a completed submission, not a pending job. Let it through.
            )
            attempt += 1

            if not response.pending:
                return response

            if on_progress is not None:
                progress = None
                if isinstance(response.body, dict):
                    progress = response.body.get("progress")
                maybe = on_progress(progress, attempt)
                if asyncio.iscoroutine(maybe):
                    await maybe

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BrainPollTimeout(
                    f"{path} still pending after {attempt} polls; giving up. "
                    "The job may still complete server-side."
                )

            delay = min(response.retry_after or self._min_retry_after, remaining)
            log.debug("brain.poll.waiting", path=path, attempt=attempt, delay=delay)
            await asyncio.sleep(delay)


def _decode(response: httpx.Response) -> Any:
    """Best-effort body decode. Some endpoints return empty bodies or non-JSON."""
    if not response.content:
        return None
    content_type = response.headers.get("content-type", "")
    if "json" not in content_type:
        return response.text
    try:
        return response.json()
    except ValueError:
        return response.text
