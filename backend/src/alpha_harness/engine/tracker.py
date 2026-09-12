"""Simulation lifecycle tracking.

The one invariant that matters:

    **A simulation id is never held only in memory.**

A ``201`` from ``POST /simulations`` returns the id *only* in the ``Location`` header,
and for a multi-simulation that id is the parent — the sole handle that can cancel the
batch. If the process dies before that id reaches disk, the simulation keeps running on
BRAIN, occupies a concurrency slot, and cannot be stopped. Preventing exactly that is
why this project exists.

So :meth:`SimulationTracker.submit` writes a row *before* the HTTP request and updates
it with the id the instant the response lands. A crash between the two leaves a
``PENDING`` row, which :meth:`reconcile` surfaces rather than silently discards.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import structlog
from sqlalchemy import func, select, update

from ..brain.endpoints import BrainEndpoints
from ..brain.errors import BrainDailyLimitReached, BrainError, BrainValidationError
from ..brain.schemas import SimulationRequest, SimulationStatus
from ..db.models import DedupEntry, QuotaSnapshot, SimStatus, SimulationRecord, utcnow
from ..db.sqlite import Database
from .dedup import hash_payload

log = structlog.get_logger(__name__)

ChangeHook = Callable[[list[dict[str, Any]]], Awaitable[None] | None]

#: How often the poll loop wakes to look for work.
TICK_SECONDS = 1.0
#: Fallback interval when the platform gives us no Retry-After.
DEFAULT_POLL_SECONDS = 3.0
#: A PENDING row older than this crashed mid-submit and needs attention.
PENDING_GRACE_SECONDS = 120.0
#: How often stale sends are looked for after startup.
SWEEP_SECONDS = 60.0

#: Platform status -> our local lifecycle. Shared with the batch engine.
STATUS_MAP = {
    SimulationStatus.COMPLETE: SimStatus.COMPLETE,
    SimulationStatus.WARNING: SimStatus.WARNING,
    SimulationStatus.ERROR: SimStatus.ERROR,
    SimulationStatus.FAIL: SimStatus.FAILED,
    SimulationStatus.CANCELLED: SimStatus.CANCELLED,
    SimulationStatus.TIMEOUT: SimStatus.TIMEOUT,
}


def extract_simulation_id(location: str | None) -> str | None:
    """Pull the id out of a ``Location`` header.

    The header is absolute (``https://api.worldquantbrain.com/simulations/{id}``) but
    tolerate a relative path too.
    """
    if not location:
        return None
    path = urlparse(location).path or location
    segment = path.rstrip("/").rsplit("/", 1)[-1]
    return segment or None


class SubmissionFailed(RuntimeError):
    """A simulation could not be started. ``record_id`` is the row that recorded it.

    ``message`` carries the platform's own words wherever it gave any — a researcher
    needs "TOP9000 is not a valid choice", not "rejected". ``fields`` keeps the
    structured form so the UI can mark the offending input.
    """

    def __init__(
        self,
        message: str,
        *,
        record_id: int | None,
        cause: Exception | None = None,
        fields: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.record_id = record_id
        self.cause = cause
        self.fields = fields or {}


class SimulationTracker:
    """Submits simulations, keeps their ids on disk, and polls them to completion."""

    def __init__(
        self,
        db: Database,
        endpoints: BrainEndpoints,
        *,
        on_change: ChangeHook | None = None,
    ) -> None:
        self.db = db
        self.endpoints = endpoints
        self._on_change = on_change
        #: Called with each new alpha id as it lands, so the vault can capture the alpha
        #: and its daily returns without anyone asking. Set by the composition root;
        #: failures inside it never affect the simulation that produced the alpha.
        self.on_alpha: Callable[[str], Awaitable[None]] | None = None
        self._task: asyncio.Task[None] | None = None
        #: Live capture tasks. Held because an unreferenced task can be collected mid
        #: flight, which would drop the returns of a completed alpha silently.
        self._captures: set[asyncio.Task[None]] = set()
        self._stopping = asyncio.Event()
        # record_id -> monotonic time of the next allowed poll. In memory only; losing
        # it just means we poll once immediately after a restart.
        self._next_poll: dict[int, float] = {}
        #: Adopted rows this process is sending right now. Their ``created_at`` is when
        #: they were queued, so without this the stale-send check could orphan one
        #: mid-request.
        self._sending: set[int] = set()
        self._last_sweep = time.monotonic()

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="simulation-tracker")
        log.info("tracker.started")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        # Alpha captures run detached, so they must be waited out here. Left running,
        # one mid-query when the database is disposed raises out of the connection pool
        # during teardown — noise that looks exactly like a real fault.
        if self._captures:
            for capture in list(self._captures):
                capture.cancel()
            await asyncio.gather(*self._captures, return_exceptions=True)
            self._captures.clear()

        log.info("tracker.stopped")

    # -- submission ------------------------------------------------------

    async def submit(
        self,
        request: SimulationRequest,
        *,
        task: str = "manual",
        record_id: int | None = None,
        user_id: str | None = None,
        endpoints: BrainEndpoints | None = None,
    ) -> SimulationRecord:
        """Start one simulation, recording its id durably. See :meth:`_submit`."""
        if record_id is not None:
            self._sending.add(record_id)
        try:
            return await self._submit(
                request, task=task, record_id=record_id, user_id=user_id, endpoints=endpoints
            )
        finally:
            if record_id is not None:
                self._sending.discard(record_id)

    async def _submit(
        self,
        request: SimulationRequest,
        *,
        task: str = "manual",
        record_id: int | None = None,
        user_id: str | None = None,
        endpoints: BrainEndpoints | None = None,
    ) -> SimulationRecord:
        payload = request.to_wire()
        settings = request.settings
        adopted = record_id is not None
        target_endpoints = endpoints or self.endpoints

        # --- 1. persist intent before touching the network ---------------
        async with self.db.session() as session:
            if record_id is not None:
                values: dict[str, Any] = {"status": SimStatus.PENDING}
                if user_id:
                    values["user_id"] = user_id
                await session.execute(
                    update(SimulationRecord)
                    .where(SimulationRecord.id == record_id)
                    .values(**values)
                )
            else:
                record = SimulationRecord(
                    request_hash=hash_payload(payload),
                    payload=payload,
                    expression=request.regular or request.combo or request.selection,
                    sim_type=str(request.type),
                    instrument_type=settings.instrument_type,
                    region=settings.region,
                    delay=settings.delay,
                    language=settings.language,
                    universe=settings.universe,
                    task=task,
                    user_id=user_id,
                    status=SimStatus.PENDING,
                )
                session.add(record)
                await session.flush()
                record_id = record.id

        log.info("sim.pending", record_id=record_id, region=settings.region, user_id=user_id)
        await self._notify()

        # --- 2. the network call ----------------------------------------
        try:
            response = await target_endpoints.create_simulation(request)
        except BrainValidationError as exc:
            # Pass the platform's own wording through untouched — "TOP9000 is not a
            # valid choice" tells the researcher what to fix; "rejected" does not.
            reason = _describe_validation(exc)
            await self._mark_rejected(record_id, reason)
            raise SubmissionFailed(
                f"BRAIN rejected these settings — {reason}",
                record_id=record_id,
                cause=exc,
                fields=exc.fields,
            ) from exc
        except BrainError as exc:
            if adopted and (exc.retryable or isinstance(exc, BrainDailyLimitReached)):
                # A throttle, an outage or the daily cap says nothing about the alpha.
                # Back in the queue, so it runs once the platform will take it.
                await self._mark(record_id, SimStatus.QUEUED, message=exc.message, finished=False)
            else:
                await self._mark_rejected(record_id, exc.message)
            raise SubmissionFailed(exc.message, record_id=record_id, cause=exc) from exc

        # --- 3. capture the id immediately ------------------------------
        platform_id = extract_simulation_id(response.location)
        if platform_id is None:
            # A 201 with no Location. We cannot cancel what we cannot name — say so
            # loudly rather than pretending the submission failed.
            await self._mark(
                record_id,
                SimStatus.ORPHANED,
                message=(
                    "BRAIN accepted the simulation but returned no Location header, so "
                    "its id is unknown and it cannot be cancelled from here. Check the "
                    "platform directly."
                ),
            )
            log.error("sim.no_location", record_id=record_id, status=response.status)
            raise SubmissionFailed(
                "Simulation started but its id was not returned; it cannot be cancelled.",
                record_id=record_id,
            )

        now = utcnow()
        async with self.db.session() as session:
            await session.execute(
                update(SimulationRecord)
                .where(SimulationRecord.id == record_id)
                .values(
                    platform_id=platform_id,
                    status=SimStatus.RUNNING,
                    submitted_at=now,
                )
            )
            if response.rate_limit is not None:
                session.add(
                    QuotaSnapshot(
                        limit_total=response.rate_limit.limit,
                        remaining=response.rate_limit.remaining,
                        reset_seconds=response.rate_limit.reset_seconds,
                    )
                )

        log.info("sim.running", record_id=record_id, platform_id=platform_id)
        self._next_poll[record_id] = time.monotonic()
        await self._notify()

        refreshed = await self.get(record_id)
        assert refreshed is not None
        return refreshed

    def watch(self, record_id: int) -> None:
        """Start polling a record immediately.

        Used by the batch engine after it records a parent id, so a freshly submitted
        batch is picked up on the next tick rather than after a delay.
        """
        self._next_poll[record_id] = time.monotonic()

    async def record_quota(self, session: Any, rate_limit: Any) -> None:
        """Store a reading of the daily quota headers, if the response carried any.

        Takes an open session so it can join the same transaction that writes the
        platform id — the quota and the id are learned from one response.
        """
        if rate_limit is None:
            return
        session.add(
            QuotaSnapshot(
                limit_total=rate_limit.limit,
                remaining=rate_limit.remaining,
                reset_seconds=rate_limit.reset_seconds,
            )
        )

    # -- cancellation ----------------------------------------------------

    async def cancel(self, record_id: int) -> bool:
        """Cancel a simulation and mark it locally.

        Returns ``False`` when there is nothing to cancel — no platform id yet, or the
        row already reached a terminal state.
        """
        record = await self.get(record_id)
        if record is None:
            return False
        if SimStatus(record.status).terminal:
            return False
        if not record.platform_id:
            # Mid-submit. Marking it here means the poll loop will not adopt it later.
            await self._mark(
                record_id,
                SimStatus.CANCELLED,
                message="Cancelled before BRAIN returned an id.",
            )
            return False

        ok = await self.endpoints.cancel_simulation(record.platform_id)
        await self._mark(
            record_id,
            SimStatus.CANCELLED,
            message=None if ok else "BRAIN did not acknowledge the cancellation.",
            finished=True,
        )
        self._next_poll.pop(record_id, None)
        log.info("sim.cancelled", record_id=record_id, platform_id=record.platform_id, ok=ok)
        return ok

    # -- reads -----------------------------------------------------------

    async def get(self, record_id: int) -> SimulationRecord | None:
        async with self.db.session() as session:
            return await session.get(SimulationRecord, record_id)

    async def active(self, user_id: str | None = None) -> list[SimulationRecord]:
        """Everything not yet in a terminal state, including work still queued.

        The queue is shown alongside what is running: a researcher needs to see the
        backlog, not just the eight slots currently in use.
        """
        async with self.db.session() as session:
            stmt = select(SimulationRecord).where(
                SimulationRecord.status.in_(
                    [SimStatus.QUEUED, SimStatus.PENDING, SimStatus.RUNNING]
                )
            )
            if user_id:
                stmt = stmt.where(SimulationRecord.user_id == user_id)
            stmt = stmt.order_by(SimulationRecord.created_at)
            result = await session.execute(stmt)
            return list(result.scalars())

    async def active_for_user(self, user_id: str | None = None) -> list[SimulationRecord]:
        return await self.active(user_id=user_id)

    async def recent(self, limit: int = 100, user_id: str | None = None) -> list[SimulationRecord]:
        async with self.db.session() as session:
            stmt = select(SimulationRecord)
            if user_id:
                stmt = stmt.where(SimulationRecord.user_id == user_id)
            stmt = stmt.order_by(SimulationRecord.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars())

    async def used_today(self, user_id: str | None = None) -> int:
        """Simulations consumed since the platform's day began.

        Counted locally because nothing exposes the real figure until a simulation POST
        comes back with its headers, and the Today page needs a number before the first
        one runs.

        Batch *parents* are excluded and their children counted instead: the platform
        counts every child of a multi-simulation, so a batch of ten costs ten. Only rows
        actually sent today count, by when they were sent: a row queued yesterday and
        run today costs today, and one dropped from the queue never cost anything.
        """
        from ..brain.filters import PLATFORM_TZ

        start = datetime.now(PLATFORM_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        async with self.db.session() as session:
            stmt = (
                select(func.count())
                .select_from(SimulationRecord)
                .where(
                    SimulationRecord.is_batch.is_(False),
                    SimulationRecord.submitted_at >= start.astimezone(UTC),
                )
            )
            if user_id:
                stmt = stmt.where(SimulationRecord.user_id == user_id)
            total = await session.scalar(stmt)
        return int(total or 0)

    async def uncharged_since(self, moment: datetime, user_id: str | None = None) -> int:
        """Simulations sent today that a quota reading taken at ``moment`` may not count.

        BRAIN's ``X-Ratelimit-Remaining`` lags what was sent: eight batches posted within
        eight seconds all read 1,337, a POST ninety seconds later read 1,327, and only a
        POST forty minutes on showed all ninety charged — cancelled ones included (seen
        live 2026-09-10). So anything still running, or that finished or was sent after
        the reading, is treated as not yet charged.

        ponytail: leans low by whatever BRAIN had already charged at the reading; drop it
        if the platform ever publishes a figure without the lag.
        """
        from sqlalchemy import or_

        from ..brain.filters import PLATFORM_TZ

        start = datetime.now(PLATFORM_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        moment = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
        async with self.db.session() as session:
            stmt = (
                select(func.count())
                .select_from(SimulationRecord)
                .where(
                    SimulationRecord.is_batch.is_(False),
                    SimulationRecord.submitted_at >= start.astimezone(UTC),
                    or_(
                        SimulationRecord.status.in_(
                            [SimStatus.QUEUED, SimStatus.PENDING, SimStatus.RUNNING]
                        ),
                        SimulationRecord.finished_at > moment,
                        SimulationRecord.submitted_at > moment,
                    ),
                )
            )
        return int(total or 0)

    async def latest_quota(self) -> QuotaSnapshot | None:
        async with self.db.session() as session:
            result = await session.execute(
                select(QuotaSnapshot).order_by(QuotaSnapshot.observed_at.desc()).limit(1)
            )
            return result.scalars().first()

    # -- recovery --------------------------------------------------------

    async def reconcile(self) -> dict[str, int]:
        """Re-adopt in-flight simulations after a restart.

        ``RUNNING`` rows have a platform id, so polling simply resumes — that is the
        common case and it is fully recoverable.

        ``PENDING`` rows are the crash window: we sent a POST but never learned the id.
        There is no reliable way to match such a request back to a simulation, so rather
        than guess, they are marked ``ORPHANED`` and reported. Being honest that a
        simulation may be running unattended is more useful than a silent cleanup.
        """
        resumed = 0
        async with self.db.session() as session:
            running = await session.execute(
                select(SimulationRecord.id).where(
                    SimulationRecord.status == SimStatus.RUNNING,
                    SimulationRecord.platform_id.is_not(None),
                )
            )
            for record_id in running.scalars():
                self._next_poll[record_id] = time.monotonic()
                resumed += 1

        # A batch's RUNNING children carry no id of their own until the parent is read
        # back. The parent's id is the handle, so they are left for the engine to expand.
        orphaned = await self.orphan_stale()

        if resumed or orphaned:
            log.warning("tracker.reconciled", resumed=resumed, orphaned=orphaned)
            await self._notify()
        return {"resumed": resumed, "orphaned": orphaned}

    async def orphan_stale(self) -> int:
        """Flag sends that never learned their id, so they stop holding a slot.

        Runs at startup and then on a timer. A send cut off by a restart can be seconds old
        when the next process starts, and checking only then would leave it PENDING, and
        counted as in flight, until the restart after.

        Judged by when the send began. A batch's children were queued long before they
        were sent, so they go by their parent, whose row is written as the batch goes out.
        Anything this process is sending right now is left alone.
        """
        cutoff = time.time() - PENDING_GRACE_SECONDS
        orphaned = 0
        async with self.db.session() as session:
            result = await session.execute(
                select(SimulationRecord).where(SimulationRecord.status == SimStatus.PENDING)
            )
            for record in result.scalars().all():
                if record.id in self._sending:
                    continue
                anchor = record
                if record.parent_record_id is not None:
                    anchor = await session.get(SimulationRecord, record.parent_record_id) or record
                created = anchor.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                if created.timestamp() >= cutoff:
                    continue
                record.status = SimStatus.ORPHANED
                record.message = (
                    "The backend stopped between sending this simulation and "
                    "recording its id. It may still be running on BRAIN and cannot "
                    "be cancelled from here — check the platform."
                )
                record.finished_at = utcnow()
                orphaned += 1

        if orphaned:
            log.warning("tracker.orphaned", count=orphaned)
            await self._notify()
        return orphaned

    # -- polling ---------------------------------------------------------

    async def _run(self) -> None:
        """Poll every active simulation, honouring each one's Retry-After."""
        while not self._stopping.is_set():
            try:
                await self._tick()
                if time.monotonic() - self._last_sweep >= SWEEP_SECONDS:
                    self._last_sweep = time.monotonic()
                    await self.orphan_stale()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("tracker.tick_failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=TICK_SECONDS)
            except TimeoutError:
                continue

    async def _tick(self) -> None:
        records = await self.active()
        if not records:
            return

        now = time.monotonic()
        changed = False
        for record in records:
            if not record.platform_id:
                continue
            if now < self._next_poll.get(record.id, 0.0):
                continue
            if await self._poll_one(record):
                changed = True

        if changed:
            await self._notify()

    async def _poll_one(self, record: SimulationRecord) -> bool:
        """One status read. Returns True if anything changed."""
        assert record.platform_id is not None
        try:
            response = await self.endpoints.client.request(
                "GET", f"/simulations/{record.platform_id}", raise_for_status=False
            )
        except BrainError as exc:
            # Transport hiccups are expected; back off and try again rather than
            # declaring a running simulation dead.
            log.warning("sim.poll_failed", record_id=record.id, error=str(exc))
            self._next_poll[record.id] = time.monotonic() + DEFAULT_POLL_SECONDS
            return False

        if response.status == 401:
            log.warning("sim.poll_unauthenticated", record_id=record.id)
            self._next_poll[record.id] = time.monotonic() + 10.0
            return False

        if response.status == 404:
            # The platform has forgotten the id; polling again cannot bring it back.
            self._next_poll.pop(record.id, None)
            await self._mark(
                record.id, SimStatus.ERROR, message="BRAIN no longer has this simulation."
            )
            return True

        if response.status >= 400:
            # A 403 or a 5xx without Retry-After is not a result. Treating it as one
            # marked running simulations COMPLETE with no alpha and stopped polling them.
            log.warning("sim.poll_error", record_id=record.id, status=response.status)
            self._next_poll[record.id] = time.monotonic() + 15.0
            return False

        body = response.body if isinstance(response.body, dict) else {}

        # Still queued or simulating: the body carries only progress.
        if response.pending:
            progress = body.get("progress")
            delay = response.retry_after or DEFAULT_POLL_SECONDS
            self._next_poll[record.id] = time.monotonic() + delay
            if progress is not None and progress != record.progress:
                async with self.db.session() as session:
                    await session.execute(
                        update(SimulationRecord)
                        .where(SimulationRecord.id == record.id)
                        .values(progress=progress, last_polled_at=utcnow())
                    )
                return True
            return False

        return await self._apply_terminal(record, body)

    async def _apply_terminal(self, record: SimulationRecord, body: dict[str, Any]) -> bool:
        """Write a finished simulation's outcome."""
        raw_status = body.get("status")
        try:
            platform_status = SimulationStatus(raw_status) if raw_status else None
        except ValueError:
            platform_status = None

        if platform_status is not None and not platform_status.terminal:
            # WAITING/SIMULATING without a Retry-After — treat as still running.
            self._next_poll[record.id] = time.monotonic() + DEFAULT_POLL_SECONDS
            return False

        local = STATUS_MAP.get(platform_status, SimStatus.COMPLETE)
        children = body.get("children") or []
        message = body.get("message")
        location = body.get("location")
        if location and not message:
            message = f"Error at line {location.get('line')}, column {location.get('start')}"

        alpha_id = body.get("alpha")

        async with self.db.session() as session:
            await session.execute(
                update(SimulationRecord)
                .where(SimulationRecord.id == record.id)
                .values(
                    status=local,
                    platform_status=str(platform_status) if platform_status else None,
                    alpha_id=alpha_id,
                    child_ids=list(children),
                    is_batch=bool(children) or record.is_batch,
                    message=message,
                    progress=1.0 if local == SimStatus.COMPLETE else record.progress,
                    finished_at=utcnow(),
                    last_polled_at=utcnow(),
                )
            )

            # Remember what this payload produced. Without it, re-running an identical
            # alpha spends daily quota to recreate something that already exists.
            # Batch parents are excluded: their payload is the array, not an alpha.
            reusable = bool(alpha_id) and not children and not record.is_batch
            if reusable and await session.get(DedupEntry, record.request_hash) is None:
                session.add(DedupEntry(request_hash=record.request_hash, alpha_id=alpha_id))

        self._next_poll.pop(record.id, None)
        log.info(
            "sim.finished",
            record_id=record.id,
            status=str(local),
            alpha_id=body.get("alpha"),
        )
        if alpha_id:
            self.alpha_landed(alpha_id)
        return True

    def alpha_landed(self, alpha_id: str) -> None:
        """Hand a finished alpha to :attr:`on_alpha` without waiting on it.

        Fire and forget. Capturing an alpha's returns is a convenience for later
        analysis; making a finished simulation wait on it, or fail with it, would be the
        wrong trade. Public because a batch's children are resolved by the engine rather
        than polled here, and they need exactly the same hand-off.
        """
        if self.on_alpha is None:
            return
        capture = asyncio.create_task(self._capture(alpha_id))
        self._captures.add(capture)
        capture.add_done_callback(self._captures.discard)

    async def _capture(self, alpha_id: str) -> None:
        hook = self.on_alpha
        if hook is None:
            return
        try:
            await hook(alpha_id)
        except Exception:
            log.warning("sim.capture_failed", alpha_id=alpha_id, exc_info=True)

    # -- helpers ---------------------------------------------------------

    async def _mark(
        self,
        record_id: int,
        status: SimStatus,
        *,
        message: str | None = None,
        finished: bool = True,
    ) -> None:
        values: dict[str, Any] = {"status": status}
        if message is not None:
            values["message"] = message
        if finished:
            values["finished_at"] = utcnow()
        async with self.db.session() as session:
            await session.execute(
                update(SimulationRecord).where(SimulationRecord.id == record_id).values(**values)
            )
        await self._notify()

    async def _mark_rejected(self, record_id: int, message: str) -> None:
        await self._mark(record_id, SimStatus.REJECTED, message=message)

    async def _notify(self) -> None:
        if self._on_change is None:
            return
        try:
            records = await self.active()
            payload = [serialise(r) for r in records]
            result = self._on_change(payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            log.exception("tracker.notify_failed")


def _describe_validation(exc: BrainValidationError) -> str:
    """Flatten BRAIN's nested field errors into one readable line."""
    parts: list[str] = []

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        elif isinstance(node, list):
            for item in node:
                parts.append(f"{prefix}: {item}" if prefix else str(item))
        else:
            parts.append(f"{prefix}: {node}" if prefix else str(node))

    walk("", exc.fields)
    return "; ".join(parts) or exc.message


def serialise(record: SimulationRecord) -> dict[str, Any]:
    """Wire shape for the API and the live WebSocket feed."""
    return {
        "id": record.id,
        "platformId": record.platform_id,
        "alphaId": record.alpha_id,
        "status": record.status,
        "platformStatus": record.platform_status,
        "progress": record.progress,
        "message": record.message,
        "expression": record.expression,
        "task": record.task,
        "region": record.region,
        "delay": record.delay,
        "universe": record.universe,
        "instrumentType": record.instrument_type,
        "language": record.language,
        "simType": record.sim_type,
        "isBatch": record.is_batch,
        "childIds": record.child_ids or [],
        # The batch parent's record id, so a child stays tied to its batch even after the
        # parent has finished and left the active set.
        "parentId": record.parent_record_id,
        "createdAt": _iso(record.created_at),
        "submittedAt": _iso(record.submitted_at),
        "finishedAt": _iso(record.finished_at),
        "elapsedSeconds": record.elapsed_seconds,
        "settings": (record.payload or {}).get("settings"),
    }


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()
