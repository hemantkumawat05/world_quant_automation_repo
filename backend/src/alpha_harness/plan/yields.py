"""Yield Rate: how much of the allowance each lab turns into submittable alphas.

The research notes give this product its objective function directly::

    Y = submittable alphas / simulated alphas,   target Y >= 0.1%

At the full daily allowance that is five submittable alphas a day, which is the
submission target with margin left for correlation filtering. A yield far below 0.1% is
the search failing, not the market being hard.

**Why yield and not Sharpe.** A lab returning three brilliant alphas from two thousand
simulations is worse, to a consultant with a fixed daily allowance, than one returning
six adequate alphas from one thousand. Ranking labs by the quality of their best result
would fund the expensive one forever. Ranking by yield funds the one that converts
compute, which is the only resource actually being spent.

This module measures; it does not judge. It joins what was simulated (SQLite, which
knows the task each simulation belonged to) against what came back (DuckDB, which knows
whether every submission check passed). The PM reads the result and allocates cores.

**On honesty of the denominator.** Only *finished* simulations count. Work still queued
has not had its chance yet, and counting it would make every lab look worse the moment
it was funded — which would teach the PM to defund whatever it just funded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import structlog
from sqlalchemy import select

from ..db.duck import Catalog
from ..db.models import SimStatus, SimulationRecord, utcnow
from ..db.sqlite import Database

log = structlog.get_logger(__name__)

#: The standard from the research notes. Below this, the search is not working.
TARGET_YIELD = 0.001

#: Checks that label an alpha rather than gate it. The platform reports these as
#: ``WARNING`` on nearly every alpha — measured live, including ones that pass every
#: gate — so judging them would make nothing submittable. Whether a candidate matches a
#: competition, theme or cluster says nothing about whether the search is working.
#:
#: ``SELF_CORRELATION`` is deliberately *not* here. An alpha too close to the pool
#: genuinely cannot be submitted, so it is a real failure to produce something new — and
#: it is precisely the failure the Diversify lab exists to answer. Excusing it would hide
#: the signal that should reallocate cores.
IGNORED_CHECKS = frozenset(
    {
        "MATCHES_COMPETITION",
        "MATCHES_PYRAMID",
        "MATCHES_THEMES",
        "CLUSTER_TEST",
        "OSMOSIS_ALLOCATION",
        "POWER_POOL_DESCRIPTION_LENGTH",
        "POWER_POOL_DESCRIPTION_FORMAT",
    }
)

#: Results that count against an alpha. On a gating check the platform reports a miss as
#: ``WARNING`` straight after simulation and as ``FAIL`` once the checks are finished, so
#: both mean the same thing.
FAILING = frozenset({"FAIL", "WARNING"})


@dataclass(frozen=True, slots=True)
class LabYield:
    """One lab's record over a window."""

    lab: str
    simulated: int
    finished: int
    alphas: int
    submittable: int
    #: Best Sharpe seen, for display only. Never for allocation — see the module note.
    best_sharpe: float | None = None

    @property
    def yield_rate(self) -> float:
        return self.submittable / self.finished if self.finished else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lab": self.lab,
            "simulated": self.simulated,
            "finished": self.finished,
            "alphas": self.alphas,
            "submittable": self.submittable,
            "yieldRate": round(self.yield_rate, 5),
            "meetsTarget": self.yield_rate >= TARGET_YIELD,
            "bestSharpe": self.best_sharpe,
            # Enough evidence to act on? One good simulation is not a trend, and the PM
            # must not defund a lab on a sample of five.
            "confident": self.finished >= 200,
        }


def judged_results(checks_json: str | None) -> list[str] | None:
    """The result of every check that describes the alpha itself, upper-cased.

    ``None`` when there is nothing to judge — a missing, unparseable or empty array, or
    one containing only competition checks. Callers treat that as "not shown to be
    good", never as "fine".
    """
    if not checks_json:
        return None
    try:
        checks = json.loads(checks_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(checks, list) or not checks:
        return None

    judged = [
        str(c.get("result", "")).upper()
        for c in checks
        if isinstance(c, dict) and str(c.get("name", "")).upper() not in IGNORED_CHECKS
    ]
    return judged or None


def is_submittable(checks_json: str | None) -> bool:
    """Whether every check that describes the alpha itself passed.

    A missing or unparseable check array is *not* submittable: the alpha has not been
    shown to be good, and assuming otherwise inflates every yield figure in the product.
    """
    results = judged_results(checks_json)
    return results is not None and all(r == "PASS" for r in results)


def is_promising(checks_json: str | None) -> bool:
    """Whether the platform is worth asking to finish judging this alpha.

    A finished simulation comes back with ``SELF_CORRELATION``, ``PROD_CORRELATION``,
    ``REGULAR_SUBMISSION`` and ``IS_LADDER_SHARPE`` still ``PENDING`` — the platform
    computes those on demand, through ``GET /alphas/{id}/check``. Until that runs, an
    alpha that will turn out to be submittable is indistinguishable from one that will
    not, and :func:`is_submittable` says no to both.

    So something has to ask. Asking about every finished alpha would be thousands of
    requests a day; asking about the ones where nothing has failed *yet* is a few dozen,
    and it is exactly the set that could still become submittable.
    """
    results = judged_results(checks_json)
    if results is None or "PENDING" not in results:
        return False
    return set(results) <= {"PASS", "PENDING"}


class YieldBook:
    """What each lab produced, per unit of allowance spent."""

    def __init__(self, db: Database, catalog: Catalog) -> None:
        self.db = db
        self.catalog = catalog

    async def by_lab(self, *, days: int = 14) -> dict[str, LabYield]:
        """Every lab's yield over the recent window, keyed by lab id."""
        since = utcnow() - timedelta(days=days)

        async with self.db.session() as session:
            rows = (
                await session.execute(
                    select(
                        SimulationRecord.task,
                        SimulationRecord.status,
                        SimulationRecord.alpha_id,
                    ).where(SimulationRecord.created_at >= since)
                )
            ).all()

        per_lab: dict[str, dict[str, Any]] = {}
        for task, status, alpha_id in rows:
            lab = lab_of(str(task))
            entry = per_lab.setdefault(lab, {"simulated": 0, "finished": 0, "alphas": set()})
            entry["simulated"] += 1
            if SimStatus(status).terminal:
                entry["finished"] += 1
            if alpha_id:
                entry["alphas"].add(str(alpha_id))

        every_alpha = sorted({a for e in per_lab.values() for a in e["alphas"]})
        verdicts = await self._verdicts(every_alpha)

        book: dict[str, LabYield] = {}
        for lab, entry in per_lab.items():
            ids = entry["alphas"]
            sharpes = [
                verdicts[a]["sharpe"] for a in ids if a in verdicts and verdicts[a]["sharpe"]
            ]
            book[lab] = LabYield(
                lab=lab,
                simulated=entry["simulated"],
                finished=entry["finished"],
                alphas=len(ids),
                submittable=sum(1 for a in ids if verdicts.get(a, {}).get("submittable")),
                best_sharpe=max(sharpes) if sharpes else None,
            )
        return book

    async def summary(self, *, days: int = 14) -> dict[str, Any]:
        """The whole book, plus the one number this product is judged on."""
        book = await self.by_lab(days=days)
        finished = sum(y.finished for y in book.values())
        submittable = sum(y.submittable for y in book.values())
        overall = submittable / finished if finished else 0.0

        return {
            "days": days,
            "labs": [y.to_dict() for y in sorted(book.values(), key=lambda y: -y.yield_rate)],
            "simulated": sum(y.simulated for y in book.values()),
            "finished": finished,
            "submittable": submittable,
            "yieldRate": round(overall, 5),
            "targetYield": TARGET_YIELD,
            "meetsTarget": overall >= TARGET_YIELD,
            # Said as a count rather than a rate: "4 of your last 1,000 were good enough"
            # is legible to someone who has never met a percentage they trusted.
            "plainly": _plainly(finished, submittable),
        }

    async def submittable(
        self,
        *,
        region: str | None = None,
        delay: int | None = None,
        universe: str | None = None,
        instrument_type: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Every alpha that passed all of BRAIN's submission checks, best Sharpe first.

        The one screen a consultant actually came for. Alphas already submitted are left
        out — they are on the platform's own list, and re-offering them is how someone
        submits the same idea twice.
        """
        clauses = ["(a.status IS NULL OR a.status = 'UNSUBMITTED')"]
        params: list[Any] = []
        for column, value in (
            ("region", region),
            ("delay", delay),
            ("universe", universe),
            ("instrument_type", instrument_type),
        ):
            if value is not None:
                clauses.append(f"a.{column} = ?")
                params.append(value)

        rows = await self.catalog.query(
            f"""
            SELECT a.* FROM alpha a
            WHERE {" AND ".join(clauses)} AND a.checks IS NOT NULL
            ORDER BY a.sharpe DESC NULLS LAST
            LIMIT ?
            """,
            [*params, max(limit * 20, 500)],
        )

        ready = [r for r in rows if is_submittable(r.get("checks"))]
        pending = sum(1 for r in rows if is_promising(r.get("checks")))
        near = sum(1 for r in rows if _near_miss(r.get("checks")))

        chosen = ready[:limit]
        labs = await self._labs_for([str(r["alpha_id"]) for r in chosen])
        series = await self._series_for([str(r["alpha_id"]) for r in chosen])

        return {
            "alphas": [
                {
                    "alphaId": str(row["alpha_id"]),
                    "expression": row.get("expression"),
                    "lab": labs.get(str(row["alpha_id"])),
                    "dateCreated": (
                        row["date_created"].isoformat() if row.get("date_created") else None
                    ),
                    "settings": {
                        "region": row.get("region"),
                        "universe": row.get("universe"),
                        "delay": row.get("delay"),
                        "neutralization": row.get("neutralization"),
                        "decay": row.get("decay"),
                        "truncation": row.get("truncation"),
                    },
                    "sharpe": row.get("sharpe"),
                    "fitness": row.get("fitness"),
                    "turnover": row.get("turnover"),
                    "returns": row.get("returns"),
                    "drawdown": row.get("drawdown"),
                    "margin": row.get("margin"),
                    "checks": json.loads(row["checks"]) if row.get("checks") else [],
                    "pnl": series.get(str(row["alpha_id"]), []),
                    "brainUrl": f"{PLATFORM_ALPHA_URL}{row['alpha_id']}",
                }
                for row in chosen
            ],
            "total": len(ready),
            #: Still being judged by the platform. Shown so an empty list reads as
            #: "not yet" rather than "never".
            "pending": pending,
            #: Failing exactly one fixable check — what the Repair lab exists for.
            "nearMisses": near,
        }

    async def _labs_for(self, alpha_ids: list[str]) -> dict[str, str]:
        """Which lab produced each alpha, read back off the task name."""
        if not alpha_ids:
            return {}
        async with self.db.session() as session:
            rows = (
                await session.execute(
                    select(SimulationRecord.alpha_id, SimulationRecord.task).where(
                        SimulationRecord.alpha_id.in_(alpha_ids)
                    )
                )
            ).all()
        return {str(a): lab_of(str(t)) for a, t in rows if a}

    async def _series_for(self, alpha_ids: list[str]) -> dict[str, list[float]]:
        """A cumulative daily PnL curve per alpha, thinned to something a sparkline can
        draw. The full series is thousands of points and no chart shows that many."""
        if not alpha_ids:
            return {}
        placeholders = ", ".join("?" for _ in alpha_ids)
        rows = await self.catalog.query(
            f"""
            SELECT alpha_id, pnl FROM alpha_pnl
            WHERE alpha_id IN ({placeholders})
            ORDER BY alpha_id, date
            """,
            list(alpha_ids),
        )
        grouped: dict[str, list[float]] = {}
        for row in rows:
            grouped.setdefault(str(row["alpha_id"]), []).append(float(row["pnl"] or 0.0))
        return {alpha_id: _sparkline(values) for alpha_id, values in grouped.items()}

    async def _verdicts(self, alpha_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Look up whether each alpha passed its checks. Chunked: DuckDB takes the ids
        as literal parameters and a day's work can be thousands of them."""
        if not alpha_ids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(alpha_ids), 500):
            chunk = alpha_ids[start : start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            rows = await self.catalog.query(
                f"SELECT alpha_id, checks, sharpe FROM alpha WHERE alpha_id IN ({placeholders})",
                chunk,
            )
            for row in rows:
                out[str(row["alpha_id"])] = {
                    "submittable": is_submittable(row.get("checks")),
                    "sharpe": row.get("sharpe"),
                }
        return out


#: Checks a near miss can plausibly be repaired for. Sharpe is not among them: a signal
#: that is not there cannot be rewritten into one, and the Repair lab would waste the
#: allowance rediscovering that. The sign flip is handled separately, on Sharpe's value
#: rather than on its check.
FIXABLE_CHECKS = frozenset(
    {
        "HIGH_TURNOVER",
        "LOW_TURNOVER",
        "LOW_FITNESS",
        "CONCENTRATED_WEIGHT",
        "LOW_SUB_UNIVERSE_SHARPE",
        "SELF_CORRELATION",
    }
)

#: Where an alpha lives on the platform. The consultant submits it there, never here.
PLATFORM_ALPHA_URL = "https://platform.worldquantbrain.com/alpha/"

#: Points in a sparkline. More than this is invisible at the size it is drawn.
SPARK_POINTS = 120


def failed_checks(checks_json: str | None) -> set[str]:
    """The names of the checks that failed, in the platform's own spelling.

    Empty when nothing failed *and* when there is nothing to read, which is why callers
    pair it with :func:`judged_results` rather than treating an empty set as a pass.
    """
    if not checks_json:
        return set()
    try:
        checks = json.loads(checks_json)
    except (json.JSONDecodeError, TypeError):
        return set()
    if not isinstance(checks, list):
        return set()

    return {
        str(c.get("name", "")).upper()
        for c in checks
        if isinstance(c, dict)
        and str(c.get("result", "")).upper() in FAILING
        and str(c.get("name", "")).upper() not in IGNORED_CHECKS
    }


def _near_miss(checks_json: str | None) -> bool:
    """One or two fixable failures and nothing else wrong."""
    failed = failed_checks(checks_json)
    return bool(failed) and failed <= FIXABLE_CHECKS and len(failed) <= 2


def _sparkline(values: list[float]) -> list[float]:
    """Daily PnL as a cumulative curve, thinned by taking every nth point."""
    if not values:
        return []
    total = 0.0
    cumulative = []
    for value in values:
        total += value
        cumulative.append(round(total, 2))
    if len(cumulative) <= SPARK_POINTS:
        return cumulative
    step = len(cumulative) / SPARK_POINTS
    thinned = [cumulative[int(i * step)] for i in range(SPARK_POINTS)]
    # Keep the last point whatever the arithmetic does: the end of the curve is the
    # number the reader actually looks at.
    thinned[-1] = cumulative[-1]
    return thinned


def lab_of(task: str) -> str:
    """Which lab a task name belongs to.

    Task names carry their lab and their day (``sweep-2026-09-08-1``), because quotas and
    progress are keyed by task while yield is judged per lab across many days.
    """
    head = task.split("-", 1)[0]
    return head or "manual"


def _plainly(finished: int, submittable: int) -> str:
    if not finished:
        return "Nothing has finished running yet, so there is nothing to judge."
    if not submittable:
        return (
            f"None of your last {finished:,} finished simulations were good enough to "
            "submit. That is normal early on — it takes volume before the good ones appear."
        )
    return (
        f"{submittable:,} of your last {finished:,} finished simulations were good enough "
        "to submit."
    )
