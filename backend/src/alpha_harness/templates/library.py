"""Storing templates.

YAML is the stored form, not the parsed model. A template is something a person edits,
and round-tripping it through a data model would discard their comments and reorder
their keys — so the source text is authoritative and ``parsed`` is a cache kept beside
it for listing and search.

The library ships with starter templates. An empty Template Studio is a blank page, and
this application is meant to be usable by someone who has never written a Fast
Expression; every seeded template is a real, documented pattern with its economic
reasoning written into the description.
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select

from ..db.models import Template
from ..db.sqlite import Database
from .schema import TemplateSpec, parse

log = structlog.get_logger(__name__)


class TemplateNotFoundError(Exception):
    def __init__(self, ref: int | str) -> None:
        super().__init__(f"No template {ref!r}.")
        self.ref = ref


class DuplicateTemplateNameError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"A template named {name!r} already exists. Rename one of them — the name is "
            "how a study refers back to the template it searched."
        )
        self.name = name


def serialise(row: Template, *, include_source: bool = True) -> dict[str, Any]:
    """A template as the API returns it."""
    payload: dict[str, Any] = {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "origin": row.origin,
        "tags": list(row.tags or []),
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "updatedAt": row.updated_at.isoformat() if row.updated_at else None,
    }
    parsed = row.parsed or {}
    payload["expr"] = parsed.get("expr")
    payload["variables"] = sorted((parsed.get("vars") or {}).keys())
    payload["settings"] = parsed.get("settings") or {}
    if include_source:
        payload["source"] = row.source
    return payload


class TemplateLibrary:
    """CRUD over stored templates."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def list(
        self,
        *,
        tag: str | None = None,
        origin: str | None = None,
        user_id: str | None = None,
    ) -> list[Template]:
        from sqlalchemy import or_

        async with self.db.session() as session:
            statement = select(Template).order_by(Template.updated_at.desc())
            if origin:
                statement = statement.where(Template.origin == origin)
            if user_id:
                statement = statement.where(
                    or_(
                        Template.user_id == user_id,
                        Template.user_id.is_(None),
                        Template.origin == "starter",
                    )
                )
            rows = list((await session.scalars(statement)).all())
        if tag:
            rows = [r for r in rows if tag in (r.tags or [])]
        return rows

    async def get(self, template_id: int) -> Template | None:
        async with self.db.session() as session:
            return await session.get(Template, template_id)

    async def by_name(self, name: str) -> Template | None:
        async with self.db.session() as session:
            return await session.scalar(select(Template).where(Template.name == name))

    async def create(
        self,
        source: str,
        *,
        origin: str = "human",
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> Template:
        spec = parse(source)
        async with self.db.session() as session:
            query = select(Template).where(Template.name == spec.name)
            if user_id:
                query = query.where(Template.user_id == user_id)
            existing = await session.scalar(query)
            if existing is not None:
                raise DuplicateTemplateNameError(spec.name)
            row = Template(
                name=spec.name,
                description=spec.description,
                source=source,
                parsed=spec.model_dump(mode="json", exclude_none=True),
                origin=origin,
                tags=list(tags or []),
                user_id=user_id,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
        log.info("template.created", name=spec.name, origin=origin, user_id=user_id)
        return row

    async def spec(self, template_id: int) -> TemplateSpec:
        row = await self.get(template_id)
        if row is None:
            raise TemplateNotFoundError(template_id)
        return parse(row.source)

    # -- seeding ---------------------------------------------------------

    async def count(self) -> int:
        async with self.db.session() as session:
            return int(await session.scalar(select(func.count()).select_from(Template)) or 0)

    async def seed(self, *, force: bool = False) -> int:
        """Install the starter templates, once.

        Skipped if the library already has anything, so a user's own work is never
        overwritten by a restart. ``force`` replaces only the seeded ones.
        """
        if force:
            async with self.db.session() as session:
                await session.execute(sql_delete(Template).where(Template.origin == "starter"))
                await session.commit()
        elif await self.count() > 0:
            return 0

        installed = 0
        for source in STARTERS:
            try:
                await self.create(source, origin="starter", tags=["starter"])
                installed += 1
            except DuplicateTemplateNameError:
                continue
            except ValueError:
                log.warning("template.seed_invalid", exc_info=True)
        log.info("template.seeded", count=installed)
        return installed


# --- starter templates ----------------------------------------------------
#
# Each is a documented pattern rather than a demo. The descriptions carry the economic
# reasoning because the point of this application is that someone who cannot yet write
# a Fast Expression can still tell whether an idea is worth testing.

STARTERS: list[str] = [
    """
name: Short-term price reversal
description: >-
  Stocks that fell over the last few days tend to bounce, and stocks that jumped tend to
  give some back. The minus sign is the whole idea: buy what went down. This is the
  simplest real alpha on the platform and a good first thing to run.
expr: rank(-ts_delta(close, $lookback))
vars:
  lookback:
    type: int
    description: Trading days of price change to react to.
    grid: [2, 3, 5, 10]
settings:
  region: USA
  delay: 1
  universe: [TOP3000, TOP1000]
  neutralization: [SUBINDUSTRY, INDUSTRY]
  decay: [0, 4]
  truncation: 0.08
  pasteurization: 'ON'
  nanHandling: 'ON'
constraints:
  max_simulations: 200
""".strip(),
    """
name: Reversal, but only when people are trading
description: >-
  The same reversal idea, held only on days when volume beats its own 20-day average.
  A price move on thin volume is noise; the condition asks the alpha to stay flat rather
  than trade a move nobody participated in.
expr: rank(trade_when(volume > adv20, -ts_delta(close, $lookback), -1))
vars:
  lookback:
    type: int
    grid: [2, 5, 10]
settings:
  region: USA
  delay: 1
  universe: [TOP3000]
  neutralization: [SUBINDUSTRY, INDUSTRY, MARKET]
  decay: [0, 6, 12]
  truncation: 0.08
constraints:
  max_simulations: 200
""".strip(),
    """
name: Fundamental ratio, ranked within industry
description: >-
  Take one accounting number, compare each company against its own history, then strip
  out whatever the whole industry did. What is left is the company being unusual for
  itself — which is the part that might be worth trading. ts_backfill fills the gaps
  between quarterly reports and does not count against the operator budget.
expr: group_neutralize(ts_zscore(ts_backfill($field, 120), $window), $group)
vars:
  field:
    type: datafield
    description: Any fundamental field with wide coverage and few alphas built on it.
    category: fundamental
    min_coverage: 0.9
    max_alpha_count: 200
    order_by: coverage
    limit: 10
  window:
    type: int
    description: >-
      Conventional windows only. Searching 37 alongside 21 is how a template becomes
      overfitted rather than informative.
    grid: [63, 252, 504]
  group:
    type: choice
    values: [industry, subindustry, sector]
settings:
  region: USA
  delay: 1
  universe: [TOP3000]
  neutralization: [INDUSTRY, SUBINDUSTRY]
  decay: [0, 8]
  truncation: 0.08
constraints:
  max_simulations: 300
""".strip(),
    """
name: Analyst estimate momentum
description: >-
  When analysts keep revising their forecasts in one direction, the price often has not
  caught up yet. ts_rank asks where today's estimate sits inside its own recent range,
  so the signal is about the direction of revisions rather than the level of the number.
expr: group_neutralize(ts_rank(ts_backfill($field, 60), $window), industry)
vars:
  field:
    type: datafield
    category: analyst
    min_coverage: 0.7
    order_by: coverage
    limit: 10
  window:
    type: int
    grid: [63, 126, 252]
settings:
  region: USA
  delay: 1
  universe: [TOP3000, TOP1000]
  neutralization: [INDUSTRY, SUBINDUSTRY]
  decay: [0, 10]
  truncation: 0.08
constraints:
  max_simulations: 200
""".strip(),
    """
name: Power Pool candidate
description: >-
  Deliberately small. Power Pool accepts at most eight operators and three distinct data
  fields, so this template is written to stay inside those limits and the constraint
  block enforces it — anything that would not qualify is rejected before it spends any
  of the daily simulation quota.
expr: group_neutralize(winsorize(ts_zscore(ts_backfill($field, 120), $window), std=4), industry)
vars:
  field:
    type: datafield
    category: fundamental
    min_coverage: 0.95
    max_alpha_count: 100
    limit: 10
  window:
    type: int
    grid: [126, 252, 504]
settings:
  region: USA
  delay: 1
  universe: [TOP3000]
  neutralization: [SUBINDUSTRY]
  decay: [0, 6]
  truncation: 0.08
constraints:
  power_pool: true
  max_simulations: 100
""".strip(),
]
