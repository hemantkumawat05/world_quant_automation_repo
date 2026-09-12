"""Operational state, stored in SQLite.

Everything here is small, transactional, and must survive a crash. The bulk analytical
data — the data-field catalog — lives in DuckDB instead (see :mod:`.duck`).

Schema evolution: additive only, applied by :mod:`.migrate` on every startup. It adds
missing columns and indexes as well as missing tables — ``create_all`` alone does *not*,
and the gap was a real bug: a column added to a model after a database was created never
appeared, and the failure surfaced much later as ``no such column`` from whichever query
selected it first.

Never drop or rename a column on ``simulation_record``. Losing a row there means losing
the ability to cancel a running simulation, which is the exact failure this project
exists to prevent. A change that is not additive is out of scope for the migrator by
design; it reports the drift rather than acting on it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


# --- credentials & session ------------------------------------------------


class Credential(Base):
    """A BRAIN login. The password is sealed by :class:`~alpha_harness.security.vault.Vault`."""

    __tablename__ = "credential"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    sessions: Mapped[list[BrainSessionRow]] = relationship(
        back_populates="credential", cascade="all, delete-orphan"
    )


class BrainSessionRow(Base):
    """A cached cookie jar, so a restart does not cost another proof-of-work solve.

    Cookies are bearer credentials, so the jar is sealed just like the password.
    """

    __tablename__ = "brain_session"

    id: Mapped[int] = mapped_column(primary_key=True)
    credential_id: Mapped[int] = mapped_column(ForeignKey("credential.id", ondelete="CASCADE"))
    cookies_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    user_id: Mapped[str | None] = mapped_column(String(64))
    permissions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    credential: Mapped[Credential] = relationship(back_populates="sessions")


# --- simulations ----------------------------------------------------------


class SimStatus(StrEnum):
    """Local lifecycle. Distinct from the platform's own status.

    Two of these carry weight:

    ``QUEUED`` — accepted locally, waiting for a free slot. Nothing has been sent to
    BRAIN, so a queued row can be dropped or re-queued freely.

    ``PENDING`` — the request is in flight and we do not yet know the platform id. This
    is the crash window: a row stuck here after a restart may correspond to a real
    running simulation, and must be reconciled rather than discarded.
    """

    QUEUED = "QUEUED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    ORPHANED = "ORPHANED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"

    @property
    def terminal(self) -> bool:
        return self not in (SimStatus.QUEUED, SimStatus.PENDING, SimStatus.RUNNING)


class SimulationRecord(Base):
    """One simulation we asked BRAIN to run.

    The row is written **before** the HTTP request and updated with ``platform_id`` the
    moment a ``201`` lands. That ordering is what guarantees a simulation can always be
    cancelled: the id is never held only in memory.

    For a multi-simulation, ``platform_id`` is the *parent* id — the only cancellable
    handle. Child ids only exist once the parent completes and are stored in
    ``child_ids``.
    """

    __tablename__ = "simulation_record"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    # Platform identity
    platform_id: Mapped[str | None] = mapped_column(String(64), index=True)
    alpha_id: Mapped[str | None] = mapped_column(String(64), index=True)
    parent_record_id: Mapped[int | None] = mapped_column(
        ForeignKey("simulation_record.id", ondelete="SET NULL")
    )
    child_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    is_batch: Mapped[bool] = mapped_column(default=False)
    #: Child ids exist only once the parent finishes, and resolving them costs one
    #: request each. This marks that work as done so it happens exactly once.
    children_expanded: Mapped[bool] = mapped_column(default=False)

    # What we asked for
    request_hash: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    expression: Mapped[str | None] = mapped_column(Text)

    # Batch-packing key: children of one multi-simulation must agree on all of these.
    sim_type: Mapped[str] = mapped_column(String(16), default="REGULAR")
    instrument_type: Mapped[str] = mapped_column(String(16), default="EQUITY")
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(16), default="FASTEXPR")
    universe: Mapped[str | None] = mapped_column(String(32))

    # Where it came from — lets the UI group work and apply per-task slot quotas.
    task: Mapped[str] = mapped_column(String(64), default="manual", index=True)
    slot: Mapped[int | None] = mapped_column(Integer)
    slot_index: Mapped[int | None] = mapped_column(Integer)

    # Lifecycle
    status: Mapped[str] = mapped_column(String(16), default=SimStatus.PENDING, index=True)
    platform_status: Mapped[str | None] = mapped_column(String(16))
    progress: Mapped[float | None] = mapped_column(Float)
    message: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_sim_active", "status", "created_at"),
        Index("ix_sim_batch_key", "sim_type", "instrument_type", "region", "delay", "language"),
    )

    @property
    def elapsed_seconds(self) -> float | None:
        """Wall time since submission — what the matrix cell counts up."""
        if self.submitted_at is None:
            return None
        end = self.finished_at or utcnow()
        start = self.submitted_at
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        return (end - start).total_seconds()


class DedupEntry(Base):
    """Hash of a canonical simulation payload -> the alpha it produced.

    Guards the daily quota against re-simulating something already run, as recommended
    in ``docs/worldquantbrain/brain-api/how-can-you-avoid-duplicate-simulations.md``.
    """

    __tablename__ = "dedup_entry"

    request_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    alpha_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TaskQuota(Base):
    """How many of the concurrent slots a named task may hold at once.

    Research runs in parallel streams — a manual experiment, a template sweep, an
    optimizer study — and without a cap one of them starves the others. A task with no
    row here competes freely for whatever is unallocated.
    """

    __tablename__ = "task_quota"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    max_slots: Mapped[int] = mapped_column(Integer, default=1)
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class QuotaSnapshot(Base):
    """A reading of ``X-RateLimit-*`` from a simulation POST.

    Kept as a series rather than a single row so the UI can show quota burn over the day.
    """

    __tablename__ = "quota_snapshot"

    id: Mapped[int] = mapped_column(primary_key=True)
    limit_total: Mapped[int | None] = mapped_column(Integer)
    remaining: Mapped[int | None] = mapped_column(Integer)
    reset_seconds: Mapped[float | None] = mapped_column(Float)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


# --- catalog sync ---------------------------------------------------------


class SyncStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SyncRun(Base):
    """One catalog crawl of a single (instrumentType, region, delay, universe) tuple.

    ``cursor_offset`` makes the crawl resumable: an interrupted sync continues from the
    last committed page rather than restarting.
    """

    __tablename__ = "sync_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_type: Mapped[str] = mapped_column(String(16), default="EQUITY")
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))

    status: Mapped[str] = mapped_column(String(16), default=SyncStatus.RUNNING, index=True)
    phase: Mapped[str | None] = mapped_column(String(32))
    cursor_offset: Mapped[int] = mapped_column(Integer, default=0)
    #: Which dataset the field crawl is inside. Fields are fetched one dataset at a
    #: time because the platform refuses any offset at or beyond 10,000, so the whole
    #: scope cannot be paged as one list — see :mod:`alpha_harness.catalog.sync`.
    cursor_dataset: Mapped[str | None] = mapped_column(String(64))
    #: Datasets whose own field count exceeds what pagination can reach. Empty in
    #: practice, and recorded rather than ignored because the alternative is a catalog
    #: that is quietly incomplete.
    truncated_datasets: Mapped[list[Any]] = mapped_column(JSON, default=list)

    datasets_synced: Mapped[int] = mapped_column(Integer, default=0)
    fields_synced: Mapped[int] = mapped_column(Integer, default=0)
    fields_expected: Mapped[int | None] = mapped_column(Integer)
    categories_synced: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_sync_tuple", "instrument_type", "region", "delay", "universe"),)

    @property
    def tuple_key(self) -> tuple[str, str, int, str]:
        return (self.instrument_type, self.region, self.delay, self.universe)


# --- platform metadata cache ---------------------------------------------


class MetadataCache(Base):
    """Cached platform metadata (``OPTIONS /simulations``, ``/operators``).

    Refreshed on login rather than hardcoded — regions and universes change.
    """

    __tablename__ = "metadata_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --- template studio (Phase 3) -------------------------------------------


class Template(Base):
    """A YAML alpha template with typed variables."""

    __tablename__ = "template"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    parsed: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    origin: Mapped[str] = mapped_column(String(16), default="human")  # human | llm
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# --- optimization (Phase 4) ----------------------------------------------


class StudyStatus(StrEnum):
    IDLE = "IDLE"
    #: Told to run, waiting for its cores to fit in the free slots.
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

    @property
    def active(self) -> bool:
        return self is StudyStatus.RUNNING


class TrialState(StrEnum):
    """Mirrors Optuna's states, plus the two this application adds.

    ``QUEUED`` and ``RUNNING`` both mean "asked for, not yet told" as far as Optuna is
    concerned; the distinction is whether a simulation has actually started, which is
    what the UI needs to show.
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    PRUNED = "PRUNED"

    @property
    def terminal(self) -> bool:
        return self in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED)


class Study(Base):
    """One optimization run over one template.

    ``template_source`` is a *snapshot*, not a reference. A study is a record of a search
    over a particular space; if the template is edited afterwards the trials stop meaning
    anything, so the space is frozen at the moment the study is created.

    Optuna itself is rebuilt from the trial rows on demand rather than persisted
    separately. One source of truth, and the trials stay joinable to the simulations that
    produced them.
    """

    __tablename__ = "study"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(128))
    template_id: Mapped[int | None] = mapped_column(ForeignKey("template.id", ondelete="SET NULL"))
    template_name: Mapped[str | None] = mapped_column(String(128))
    template_source: Mapped[str] = mapped_column(Text)

    sampler: Mapped[str] = mapped_column(String(32), default="nsga3")
    sampler_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sampler_notes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    objectives: Mapped[list[Any]] = mapped_column(JSON, default=list)
    directions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    seed: Mapped[int | None] = mapped_column(Integer)

    #: Trials asked for per round. Kept a multiple of ten so a round fills whole
    #: multi-simulations rather than leaving slots holding part-empty batches.
    batch_size: Mapped[int] = mapped_column(Integer, default=10)
    max_trials: Mapped[int] = mapped_column(Integer, default=80)
    #: Slot-quota group, so a study cannot starve manual work.
    task: Mapped[str] = mapped_column(String(64), default="study")

    status: Mapped[str] = mapped_column(String(16), default=StudyStatus.IDLE, index=True)
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    trials: Mapped[list[Trial]] = relationship(back_populates="study", cascade="all, delete-orphan")


class Trial(Base):
    """One point in the search space, and what it turned into.

    The link to ``simulation_record`` is the whole reason trials live here rather than in
    Optuna's own storage: "which parameters produced this alpha" and "which simulation is
    this trial waiting on" are the two questions actually asked of a running study.
    """

    __tablename__ = "trial"

    id: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("study.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)

    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    distributions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    expression: Mapped[str | None] = mapped_column(Text)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    state: Mapped[str] = mapped_column(String(16), default=TrialState.QUEUED, index=True)
    values: Mapped[list[Any] | None] = mapped_column(JSON)
    #: Submission-check violations, keyed by the platform's own check name. Zero or
    #: less is feasible, following Optuna's convention.
    constraint: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    feasible: Mapped[bool | None] = mapped_column()
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    message: Mapped[str | None] = mapped_column(Text)

    simulation_record_id: Mapped[int | None] = mapped_column(
        ForeignKey("simulation_record.id", ondelete="SET NULL"), index=True
    )
    alpha_id: Mapped[str | None] = mapped_column(String(64), index=True)
    #: Evolution Lab only: 0 for the seeds, then one per bred generation.
    generation: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    study: Mapped[Study] = relationship(back_populates="trials")

    __table_args__ = (
        UniqueConstraint("study_id", "number", name="uq_trial_number"),
        Index("ix_trial_open", "study_id", "state"),
    )


# --- conversations --------------------------------------------------------


class ChatThread(Base):
    """One conversation with the assistant.

    Persisted because a consultant who closes the laptop mid-thought should not lose it,
    and because the scope is part of the conversation: "which fields" means nothing
    without knowing which market is being talked about.
    """

    __tablename__ = "chat_thread"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    title: Mapped[str] = mapped_column(String(160))
    instrument_type: Mapped[str] = mapped_column(String(16), default="EQUITY")
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, index=True
    )

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="thread", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    """One turn. ``meta`` carries the picks, model and cost of an assistant turn."""

    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[int] = mapped_column(
        ForeignKey("chat_thread.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    text: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    thread: Mapped[ChatThread] = relationship(back_populates="messages")


# --- the day's plan -------------------------------------------------------


class PlanTrack(Base):
    """One line of research running today, holding some of the eight slots.

    A day's allowance is spent by splitting it across a few tracks that differ from each
    other, rather than pouring it all into one recipe. Two reasons, and the second is the
    one that matters: a single recipe finds one kind of alpha, and five hundred people
    running the same single recipe find the same one.

    Progress is deliberately *not* stored here. It is counted from ``simulation_record``
    by task, so it can never drift from what actually ran — the alternative is a
    percentage that keeps climbing after the work has stopped.
    """

    __tablename__ = "plan_track"
    __table_args__ = (UniqueConstraint("day", "task", name="uq_plan_track_day_task"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    #: US-Eastern date, because that is the clock the simulation allowance resets on.
    day: Mapped[str] = mapped_column(String(10), index=True)
    #: The engine task name. Slot quotas are keyed by this.
    task: Mapped[str] = mapped_column(String(64))
    #: What this track is, in words a beginner reads once and understands.
    name: Mapped[str] = mapped_column(String(160))
    explains: Mapped[str] = mapped_column(Text, default="")
    lab: Mapped[str] = mapped_column(String(32), default="explore")
    #: The arguments that generated the work, kept so the track can be explained,
    #: repeated, or told apart from another that looks similar on the surface.
    recipe: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    target: Mapped[int] = mapped_column(Integer, default=0)
    cores: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --- LLM keys (Phase 5) ---------------------------------------------------


class ApiKey(Base):
    """An assistant key. Sealed at rest; only a masked hint is ever shown.

    ``provider`` says whose key it is. Rotation is per provider: a Groq key cannot answer
    a request for a Gemini model, and offering it would spend a retry to learn that.
    """

    __tablename__ = "api_key"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    label: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32), default="google", server_default="google")
    key_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    hint: Mapped[str] = mapped_column(String(32))
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    last_ok_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KeyUsage(Base):
    """Local budget ledger per (key, model, quota day).

    Google exposes no remaining-quota endpoint, so RPM/TPM/RPD have to be tracked
    client-side or rotation is guesswork.

    The day is **America/Los_Angeles**, which is where AI Studio's quota clock lives.
    Counting UTC days would hand a key's daily budget back seven or eight hours early
    and produce 429s that look like the platform misbehaving.
    """

    __tablename__ = "key_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_key_id: Mapped[int] = mapped_column(ForeignKey("api_key.id", ondelete="CASCADE"))
    model: Mapped[str] = mapped_column(String(64))
    day: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD, America/Los_Angeles
    requests: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    last_request_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (UniqueConstraint("api_key_id", "model", "day", name="uq_key_model_day"),)
