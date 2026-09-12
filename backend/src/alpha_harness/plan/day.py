"""The day's plan: a few different lines of research, sharing the eight slots.

This is the answer to the only question the application really has to solve. A
consultant is given five thousand simulations a day and spends two hundred. The rest
evaporate at midnight. They will give this ten minutes and they do not know what a
z-score is.

So the plan is built *for* them, and their entire job is to press go. But it must not be
the **same** plan for everyone: five hundred people running one recipe produce one
discovery five hundred times over, which wastes the compute just as thoroughly as not
running it at all. Diversity therefore has to come out of the machine, not out of the
user's expertise.

It comes from two places here.

**The levers.** Four axes — how fast the signal moves, what shape it takes, whether the
data is crowded or untouched, and what it is compared against. Every combination is a
legitimate piece of research; none is a wrong answer. That is the point. Someone pressing
buttons at random still ends up somewhere useful, and two people pressing at random
almost never end up in the same place.

**The depth.** The levers alone are not enough, because the same lever settings pick the
same fields. So each track also steps a rank deeper into every dataset (see
``best_fields(skip=...)``). The second-widest field in a dataset is not worse than the
widest; it is simply *different*, which is the whole requirement.

There is deliberately no identity in any of this — nothing is derived from who you are or
what your user id hashes to. The spread comes from the size of the option space and an
ordinary random draw across it, so two consultants who happen to make identical choices
still get different work, and nobody is quietly assigned a worse slice of the search than
their neighbour.

**What actually runs it: nothing here.** The tracks are generated, queued under their own
task names, given slot quotas, and then this module is done. :class:`BatchEngine` already
drains a queue across free slots with per-task ceilings, so five thousand queued rows are
spent by machinery that exists. A plan is a *shape given to the queue*, not a scheduler.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import func, select

from ..catalog.queries import Tuple4
from ..db.models import PlanTrack, SimStatus, SimulationRecord
from ..db.sqlite import Database
from ..engine.slots import BatchEngine
from ..harvest.seeds import SeedHarvester

log = structlog.get_logger(__name__)

#: The allowance resets on this clock, not on the machine's local one.
EASTERN = ZoneInfo("America/New_York")

#: What a consultant is given each day. Overridden the moment the platform tells us
#: otherwise — a simulation POST returns the real number in ``X-Ratelimit-Limit``.
ASSUMED_DAILY = 5_000

#: More than this and the tracks stop being distinguishable to the person reading them.
MAX_TRACKS = 4


def today() -> str:
    return datetime.now(EASTERN).date().isoformat()


# -- the levers ------------------------------------------------------------
#
# Each entry is (key, the words shown, the arguments it means). The words matter as much
# as the arguments: someone who cannot read the third column has to be able to choose
# from the second and still be choosing something real.

SPEEDS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("fast", "Fast-moving", (63, 126)),
    ("slow", "Slow-moving", (252, 504)),
    ("mixed", "Mixed-speed", (126, 252)),
)

SHAPES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("unusual", "unusual numbers", ("ts_zscore_neutral", "rank_zscore")),
    ("position", "numbers near the top or bottom of their range", ("ts_rank",)),
    ("movement", "numbers that moved a lot", ("change",)),
    ("reversal", "moves that tend to snap back", ("reversal",)),
    ("size", "numbers measured against company size", ("scaled_by_cap",)),
)

CROWDING: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("untouched", "in data almost nobody has used", {"min_alphas": 1, "max_alphas": 40}),
    ("proven", "in data that has already produced alphas", {"min_alphas": 150}),
    ("open", "across all the usable data", {"min_alphas": 3}),
)

COMPARED: tuple[tuple[str, str, str], ...] = (
    ("industry", "against its industry", "industry"),
    ("subindustry", "against its closest competitors", "subindustry"),
    ("sector", "against its sector", "sector"),
    ("market", "against the whole market", "market"),
)

#: How many ranks down into each dataset a track reaches. Three tracks at three depths
#: touch three disjoint sets of fields.
DEPTHS: tuple[int, ...] = (0, 1, 2, 3)

#: One track's worth of choices, before it becomes a runnable recipe.
type Combination = tuple[
    tuple[str, str, tuple[int, ...]],
    tuple[str, str, tuple[str, ...]],
    tuple[str, str, dict[str, Any]],
    tuple[str, str, str],
    int,
]

_BY_KEY: dict[str, dict[str, Any]] = {
    "speed": {k: e for e in SPEEDS for k in (e[0],)},
    "shape": {k: e for e in SHAPES for k in (e[0],)},
    "crowding": {k: e for e in CROWDING for k in (e[0],)},
    "compared": {k: e for e in COMPARED for k in (e[0],)},
}


def resolve(choice: dict[str, Any]) -> Combination | None:
    """Turn named lever choices into a combination, or nothing if any name is wrong.

    Used for anything proposed from outside this module — the assistant especially. A
    model that invents a lever it would like to exist must not be able to smuggle
    settings past the option space, so an unrecognised name rejects the whole track
    rather than being quietly defaulted: a track silently different from the one that
    was explained to the user is worse than a track that never ran.
    """
    try:
        depth = int(choice.get("depth", 0))
    except (TypeError, ValueError):
        return None
    if depth not in DEPTHS:
        return None

    picked = [_BY_KEY[axis].get(str(choice.get(axis))) for axis in _BY_KEY]
    if any(p is None for p in picked):
        return None
    speed, shape, crowd, compared = picked
    return (speed, shape, crowd, compared, depth)


def lever_catalogue() -> dict[str, Any]:
    """The levers, for a screen that wants to show what varies."""
    return {
        "speed": [{"key": k, "label": v} for k, v, _ in SPEEDS],
        "shape": [{"key": k, "label": v} for k, v, _ in SHAPES],
        "crowding": [{"key": k, "label": v} for k, v, _ in CROWDING],
        "compared": [{"key": k, "label": v} for k, v, _ in COMPARED],
        "combinations": len(SPEEDS) * len(SHAPES) * len(CROWDING) * len(COMPARED) * len(DEPTHS),
    }


class DayPlanner:
    """Builds a day's research and hands it to the engine."""

    def __init__(self, db: Database, engine: BatchEngine, harvester: SeedHarvester) -> None:
        self.db = db
        self.engine = engine
        self.harvester = harvester

    # -- proposing -------------------------------------------------------

    def suggest(
        self,
        *,
        tracks: int = 3,
        target: int = ASSUMED_DAILY,
        slots: int | None = None,
        seed: int | None = None,
        neutralization: str = "SUBINDUSTRY",
    ) -> list[dict[str, Any]]:
        """Draw ``tracks`` different lines of research. Writes nothing.

        Nothing is persisted so that "show me a different plan" costs nothing and can be
        pressed as many times as someone likes. Re-rolling is itself one of the controls:
        it feels like choosing, and every outcome is a real piece of research.
        """
        tracks = max(1, min(tracks, MAX_TRACKS))
        target = max(1, target)
        slots = slots or self.engine.slots
        rng = random.Random(seed)

        # Vary every axis across the plan rather than drawing whole combinations at
        # random. Uniform draws over the combination space repeat individual levers
        # often — three tracks all judged "against the whole market" is a legitimate
        # plan that *reads* as one idea printed three times, and this audience decides
        # in seconds whether the app is doing anything. Shuffling each axis and dealing
        # round-robin makes the tracks differ on every lever that has options to spare,
        # which is both more useful and more obviously useful.
        drawn = list(
            zip(
                _deal(SPEEDS, tracks, rng),
                _deal(SHAPES, tracks, rng),
                _deal(CROWDING, tracks, rng),
                _deal(COMPARED, tracks, rng),
                _deal(DEPTHS, tracks, rng),
                strict=True,
            )
        )
        return self.build(drawn, target=target, slots=slots, neutralization=neutralization)

    def build(
        self,
        combinations: Sequence[Combination],
        *,
        target: int = ASSUMED_DAILY,
        slots: int | None = None,
        neutralization: str = "SUBINDUSTRY",
        reasons: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Turn lever combinations into runnable tracks.

        The one place a track is made, whether the combination came from a random draw
        or from the assistant. Anything that can propose research has to come through
        here, so a suggestion can never carry settings that a drawn track could not.
        """
        slots = slots or self.engine.slots
        share = _split(max(1, target), len(combinations))
        cores = _split(slots, len(combinations))

        proposals = []
        for index, (speed, shape, crowd, compared, depth) in enumerate(combinations):
            speed_key, speed_label, windows = speed
            shape_key, shape_label, patterns = shape
            crowd_key, crowd_label, crowd_args = crowd
            compared_key, compared_label, group = compared

            proposals.append(
                {
                    "task": f"plan-{index + 1}",
                    "name": f"{speed_label} {shape_label} {crowd_label}",
                    "explains": (
                        f"Looks for {shape_label} that show up over "
                        f"{_window_words(windows)}, {crowd_label}, and judges each one "
                        f"{compared_label}."
                    ),
                    "why": reasons[index] if reasons and index < len(reasons) else None,
                    "levers": {
                        "speed": speed_key,
                        "shape": shape_key,
                        "crowding": crowd_key,
                        "compared": compared_key,
                        "depth": depth,
                    },
                    "lab": "explore",
                    "target": share[index],
                    "cores": cores[index],
                    "recipe": {
                        "patterns": list(patterns),
                        "windows": list(windows),
                        "group": group,
                        "neutralization": neutralization,
                        "skip": depth,
                        **crowd_args,
                    },
                }
            )
        return proposals

    # -- starting --------------------------------------------------------

    async def start(
        self,
        scope: Tuple4,
        proposals: list[dict[str, Any]],
        *,
        replace: bool = True,
    ) -> dict[str, Any]:
        """Generate the work, queue it, and give each track its share of the slots."""
        if not proposals:
            raise ValueError("A plan needs at least one line of research.")

        day = today()
        if replace:
            await self.stop_all()

        # Task names must be unique for the day, not just for the plan. They key the
        # slot quotas *and* the progress count, so reusing "plan-1" after a restart would
        # both collide in storage and silently add this morning's finished simulations to
        # this afternoon's progress bar.
        offset = await self._tracks_today()

        started: list[dict[str, Any]] = []
        for index, proposal in enumerate(proposals):
            # The date is part of the name because a long plan outlives the day that
            # started it: 5,000 simulations at eight slots takes hours, so a plan begun
            # in the evening is still draining after midnight. Without the date,
            # yesterday's leftovers and today's fresh track would share a name and their
            # progress would be added together.
            task = f"plan-{day}-{offset + index + 1}"
            recipe = dict(proposal.get("recipe") or {})
            target = int(proposal.get("target") or 0)
            cores = max(1, int(proposal.get("cores") or 1))

            harvest = await self.harvester.harvest(scope, target=target, **recipe)
            if not harvest.requests:
                # A track that produced nothing usually means the levers landed on data
                # this market does not have. Say so rather than silently dropping it —
                # a plan that quietly runs two of three tracks is a plan nobody trusts.
                log.warning("plan.track_empty", task=task, scope=scope.label, recipe=recipe)
                started.append(
                    {**proposal, "task": task, "queued": 0, "skipped": 0, "empty": True, "day": day}
                )
                continue

            result = await self.engine.enqueue(harvest.requests, task=task)
            await self.engine.set_quota(task, cores)

            async with self.db.session() as session:
                session.add(
                    PlanTrack(
                        day=day,
                        task=task,
                        name=str(proposal["name"]),
                        explains=str(proposal.get("explains") or ""),
                        lab=str(proposal.get("lab") or "explore"),
                        recipe={**recipe, "levers": proposal.get("levers") or {}},
                        target=target,
                        cores=cores,
                    )
                )
                await session.commit()

            queued_here = len(result["queued"])
            if queued_here < target:
                # The market ran out of usable data before the target did. Reported
                # rather than absorbed: someone who asked for 5,000 and got 2,400 needs
                # to know the day is short, or they will believe it is covered.
                log.info(
                    "plan.track_short", task=task, asked=target, got=queued_here, scope=scope.label
                )

            started.append(
                {
                    **proposal,
                    "task": task,
                    "day": day,
                    "empty": False,
                    "queued": queued_here,
                    "short": queued_here < target,
                    "skipped": len(result["skipped"]),
                    "fields": len(harvest.fields),
                }
            )

        queued = sum(t["queued"] for t in started)
        log.info("plan.started", day=day, tracks=len(started), queued=queued)
        return {
            "day": day,
            "scope": scope.model_dump(by_alias=True),
            "tracks": started,
            "queued": queued,
            "skipped": sum(t["skipped"] for t in started),
        }

    # -- watching --------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        """The plan in progress and how far through it is.

        Selected by *state*, not by date. A plan started last night is still spending
        the allowance this morning, and filtering on today's date would leave that work
        running with nothing on screen to show for it — the one thing this application
        must never do. Each track carries the day it began, so an overnight one is
        visibly an overnight one rather than quietly mixed in with today's.
        """
        day = today()
        async with self.db.session() as session:
            rows = list(
                (
                    await session.scalars(
                        select(PlanTrack)
                        .where((PlanTrack.day == day) | (PlanTrack.status == "running"))
                        .order_by(PlanTrack.day, PlanTrack.id)
                    )
                ).all()
            )
            counts = await self._counts(session, [r.task for r in rows])

        tracks = []
        for row in rows:
            by_status = counts.get(row.task, {})
            done = sum(n for s, n in by_status.items() if SimStatus(s).terminal)
            running = sum(n for s, n in by_status.items() if not SimStatus(s).terminal)
            tracks.append(
                {
                    "task": row.task,
                    "day": row.day,
                    #: True for a plan that began before today and is still draining.
                    "fromEarlier": row.day != day,
                    "name": row.name,
                    "explains": row.explains,
                    "lab": row.lab,
                    "levers": (row.recipe or {}).get("levers", {}),
                    "target": row.target,
                    "cores": row.cores,
                    "status": row.status,
                    "done": done,
                    "waiting": by_status.get(str(SimStatus.QUEUED), 0),
                    "inFlight": running - by_status.get(str(SimStatus.QUEUED), 0),
                    "byStatus": by_status,
                }
            )

        done = sum(t["done"] for t in tracks)
        waiting = sum(t["waiting"] for t in tracks)
        return {
            "day": day,
            "active": bool(tracks),
            "tracks": tracks,
            "done": done,
            "waiting": waiting,
            "inFlight": sum(t["inFlight"] for t in tracks),
            "target": sum(t["target"] for t in tracks),
            # The number this whole application exists to move. Left as a plain count
            # rather than a percentage: "3,140 still to spend today" lands, "63%" does not.
            "remaining": max(0, sum(t["target"] for t in tracks) - done),
        }

    async def stop(self, task: str) -> int:
        """Stop one track. Anything already sent to BRAIN keeps running.

        Found by task name alone rather than by today's date: task names carry their own
        day, and a track begun last night has to remain stoppable this morning or it
        drains the allowance with no way to call it off.
        """
        dropped = await self.engine.drop_queued(task)
        await self.engine.set_quota(task, 0, enabled=False)
        async with self.db.session() as session:
            row = await session.scalar(select(PlanTrack).where(PlanTrack.task == task))
            if row is not None:
                row.status = "stopped"
                await session.commit()
        return dropped

    async def stop_all(self) -> int:
        """Stop every running track, including any left over from an earlier day."""
        async with self.db.session() as session:
            rows = list(
                (
                    await session.scalars(select(PlanTrack).where(PlanTrack.status == "running"))
                ).all()
            )
        return sum([await self.stop(row.task) for row in rows])

    async def _tracks_today(self) -> int:
        async with self.db.session() as session:
            return int(
                await session.scalar(
                    select(func.count()).select_from(PlanTrack).where(PlanTrack.day == today())
                )
                or 0
            )

    async def _counts(self, session: Any, tasks: list[str]) -> dict[str, dict[str, int]]:
        if not tasks:
            return {}
        result = await session.execute(
            select(SimulationRecord.task, SimulationRecord.status, func.count())
            .where(SimulationRecord.task.in_(tasks))
            .group_by(SimulationRecord.task, SimulationRecord.status)
        )
        counts: dict[str, dict[str, int]] = {}
        for task, status, n in result.all():
            counts.setdefault(task, {})[str(status)] = n
        return counts


def _deal[T](options: Sequence[T], count: int, rng: random.Random) -> list[T]:
    """``count`` values from one axis, all different while the options last."""
    pool = list(options)
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(count)]


def _split(total: int, parts: int) -> list[int]:
    """Divide as evenly as possible, giving the remainder to the earliest tracks."""
    if parts <= 0:
        return []
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _window_words(windows: tuple[int, ...]) -> str:
    """Trading-day windows as durations, because 504 means nothing to a beginner."""
    names = {21: "a month", 63: "three months", 126: "six months", 252: "a year", 504: "two years"}
    said = [names.get(w, f"{w} days") for w in windows]
    return said[0] if len(said) == 1 else " and ".join([", ".join(said[:-1]), said[-1]])
