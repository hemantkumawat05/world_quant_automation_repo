"""Invent: the structure of the formula itself.

Every other lab holds a shape still and varies something inside it — the field, the
window, the market, the settings. This one varies the shape. It is the widest search in
the product and the only one that can produce an idea nobody wrote down first.

**It builds expressions rather than writing them.** An alpha here is a small typed tree:
a data field, a time-series operator over it, a cross-sectional step, and a grouping. The
tree is what makes the search safe. String-level tinkering with expressions produces
things that parse and mean nothing — ``ts_rank(subindustry, 252)`` is syntactically fine
and semantically rubbish, and the platform charges the same allowance for it. Because the
tree is typed, a group can only ever land in a grouping slot and a field only in a field
slot, so everything this lab emits is well formed by construction.

**It evolves rather than restarts.** Each run reads the alphas already in this market,
scores them on the utility from the research notes — Sharpe, Fitness and Turnover, minus
a point for every check the platform failed — and builds the next batch mostly out of the
best of them: half by crossing two, most of the rest by mutating one, a few from nothing.
Nothing is re-run verbatim, and the population lives in the vault, so stopping the
application loses no progress.

**No assistant required.** The old path — asking a model for finished expressions — is
still there under Ask, and it is genuinely good at proposing shapes a grammar would not.
But it needed a key, and a lab that needs a key is a lab most people never open.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, replace
from typing import Any

import structlog

from ..brain.schemas import SimulationRequest, SimulationSettings
from ..catalog.queries import CatalogQueries, Tuple4
from ..harvest.patterns import WINDOWS
from ..harvest.seeds import SeedHarvester
from ..templates.validate import (
    POWER_POOL_MAX_FIELDS,
    POWER_POOL_MAX_OPERATORS,
    count_operators,
    operators_in,
    unique_data_fields,
)
from ..vault.store import AlphaVault

log = structlog.get_logger(__name__)

MAX_SIMULATIONS = 20_000

#: Below this a field cannot carry a signal about the market whatever else is true.
MIN_COVERAGE = 0.5

#: Every field is backfilled. Most data outside price and volume updates quarterly, and
#: ``ts_backfill`` costs nothing against the operator budget.
BACKFILL = 120

#: How many finished alphas to breed from. Beyond this the tail contributes nothing but
#: its own mediocrity.
POPULATION = 40

#: Make the signal comparable over time.
TS_OPS: tuple[str, ...] = (
    "ts_zscore",
    "ts_rank",
    "ts_delta",
    "ts_mean",
    "ts_std_dev",
    "ts_decay_linear",
)

#: Make it comparable across stocks. ``sub`` and ``mul`` take two ideas, the rest one.
CROSS_TWO: tuple[str, ...] = ("sub", "mul")
CROSS_ONE: tuple[str, ...] = ("rank", "zscore", "neg")

#: Strip out whatever the group shares, so what is left is specific to the company.
GROUP_OPS: tuple[str, ...] = ("group_neutralize", "group_rank", "group_zscore")

#: What each stock is compared against. The last is a group built out of company size,
#: which costs three operators and a field — worth it when nothing else separates them.
GROUPINGS: tuple[str, ...] = (
    "industry",
    "subindustry",
    "sector",
    'densify(bucket(rank(cap), range = "0.1, 1, 0.1"))',
)

#: How the next batch is made up. Elites are never re-issued unchanged: an alpha the
#: platform has already judged tells us nothing a second time.
CROSSOVER_SHARE = 0.50
ELITE_MUTATION_SHARE = 0.25
MUTATION_SHARE = 0.20
#: The remainder is drawn from nothing at all, which is what stops the search collapsing
#: onto one family however good that family looked early.

#: The best few, for elite mutation.
ELITES = 8

_INT = re.compile(r"^-?\d+$")


# -- the tree --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Node:
    """One step of an alpha.

    ``kind`` is the slot this node can occupy, and it is the whole safety argument:
    crossover only ever swaps nodes of the same kind, so a grouping cannot end up where a
    number belongs.
    """

    kind: str  # "field" | "ts" | "cross" | "group"
    op: str  # operator name, or the field id for a leaf
    children: tuple[Node, ...] = ()
    window: int = 0
    group: str = ""

    def render(self) -> str:
        if self.kind == "field":
            return f"ts_backfill({self.op}, {BACKFILL})"
        if self.kind == "pair":
            return f"{self.children[0].render()} * {self.children[1].render()}"
        if self.kind == "ts":
            return f"{self.op}({self.children[0].render()}, {self.window})"
        if self.kind == "group":
            return f"{self.op}({self.children[0].render()}, {self.group})"
        if self.op == "sub":
            return f"rank({self.children[0].render()}) - rank({self.children[1].render()})"
        if self.op == "mul":
            return f"rank({self.children[0].render()}) * rank({self.children[1].render()})"
        if self.op == "neg":
            return f"-({self.children[0].render()})"
        return f"{self.op}({self.children[0].render()})"

    def fields(self) -> set[str]:
        if self.kind == "field":
            return {self.op}
        return {f for child in self.children for f in child.fields()}


def paths(node: Node, prefix: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], Node]]:
    """Every node in the tree with the route to it."""
    found = [(prefix, node)]
    for index, child in enumerate(node.children):
        found.extend(paths(child, (*prefix, index)))
    return found


def swap(node: Node, path: tuple[int, ...], other: Node) -> Node:
    """The tree with one node replaced."""
    if not path:
        return other
    index, rest = path[0], path[1:]
    children = list(node.children)
    children[index] = swap(children[index], rest, other)
    return replace(node, children=tuple(children))


# -- reading a tree back out of an expression ------------------------------


def parse(expression: str | None) -> Node | None:
    """An expression turned back into a tree, or ``None`` if it is not this grammar.

    Only ever fed expressions this lab or Sweep produced, so it recognises exactly those
    forms and refuses everything else. Refusing is the safe answer: an alpha that cannot
    be read is simply not bred from.
    """
    if not expression:
        return None
    text = expression.strip()
    if ";" in text or not text:
        return None
    return _parse(text)


def _parse(text: str) -> Node | None:
    text = text.strip()
    while text.startswith("(") and _matching(text, 0) == len(text) - 1:
        text = text[1:-1].strip()
    if not text:
        return None

    if text.startswith("-"):
        child = _parse(text[1:])
        return Node("cross", "neg", (child,)) if child else None

    for symbol, op in ((" - ", "sub"), (" * ", "mul")):
        left, right = _split_binary(text, symbol)
        if left is None or right is None:
            continue
        a, b = _unrank(left), _unrank(right)
        if a is not None and b is not None:
            return Node("cross", op, (a, b))
        # Two raw fields multiplied, which is a different idea from multiplying their
        # ranks: it is the two moving together rather than the two both being high.
        a, b = _parse(left), _parse(right)
        if op == "mul" and a is not None and b is not None and a.kind == b.kind == "field":
            return Node("pair", "mul", (a, b))

    call = _split_call(text)
    if call is None:
        return None
    name, args = call

    if name == "ts_backfill" and len(args) == 2:
        return Node("field", args[0].strip())
    if name in TS_OPS and len(args) == 2 and _INT.match(args[1].strip()):
        child = _parse(args[0])
        return Node("ts", name, (child,), window=int(args[1])) if child else None
    if name in {"rank", "zscore"} and len(args) == 1:
        child = _parse(args[0])
        return Node("cross", name, (child,)) if child else None
    if name in GROUP_OPS and len(args) == 2:
        child = _parse(args[0])
        return Node("group", name, (child,), group=args[1].strip()) if child else None
    return None


def _unrank(text: str) -> Node | None:
    """The inside of a ``rank(...)``, which is how ``sub`` and ``mul`` are written."""
    call = _split_call(text.strip())
    if call is None or call[0] != "rank" or len(call[1]) != 1:
        return None
    return _parse(call[1][0])


def _split_call(text: str) -> tuple[str, list[str]] | None:
    match = re.match(r"([a-zA-Z_]\w*)\s*\(", text)
    if not match or _matching(text, match.end() - 1) != len(text) - 1:
        return None
    return match.group(1), _split_args(text[match.end() : -1])


def _split_args(text: str) -> list[str]:
    args, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:index])
            start = index + 1
    args.append(text[start:])
    return [a.strip() for a in args if a.strip()]


def _split_binary(text: str, symbol: str) -> tuple[str | None, str | None]:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and text.startswith(symbol, index):
            return text[:index], text[index + len(symbol) :]
    return None, None


def _matching(text: str, opening: int) -> int:
    """Index of the bracket closing the one at ``opening``, or -1."""
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


# -- making trees ----------------------------------------------------------


def grow(rng: random.Random, fields: list[dict[str, Any]]) -> Node | None:
    """One alpha, drawn from the grammar."""
    if not fields:
        return None
    first = rng.choice(fields)
    second = _companion(rng, fields, first) if rng.random() < 0.5 else None

    if second is not None and rng.random() < 0.5:
        # Each made comparable first, then set against the other.
        tree: Node = Node(
            "cross",
            rng.choice(CROSS_TWO),
            (_timed(rng, first), _timed(rng, second)),
        )
    elif second is not None:
        # The two multiplied and then ranked together, which fires only where both are
        # high at once. A different bet from ranking them separately and subtracting.
        pair = Node(
            "pair",
            "mul",
            (Node("field", str(first["field_id"])), Node("field", str(second["field_id"]))),
        )
        tree = Node("ts", rng.choice(TS_OPS), (pair,), window=rng.choice(WINDOWS))
    else:
        tree = _timed(rng, first)
        if rng.random() < 0.6:
            tree = Node("cross", rng.choice(CROSS_ONE), (tree,))

    if rng.random() < 0.7:
        tree = Node("group", rng.choice(GROUP_OPS), (tree,), group=rng.choice(GROUPINGS))
    return tree


def _timed(rng: random.Random, field: dict[str, Any]) -> Node:
    return Node(
        "ts",
        rng.choice(TS_OPS),
        (Node("field", str(field["field_id"])),),
        window=rng.choice(WINDOWS),
    )


def _companion(
    rng: random.Random, fields: list[dict[str, Any]], first: dict[str, Any]
) -> dict[str, Any] | None:
    """Another field to set against this one, from the same dataset where possible.

    Preferred rather than required. Using one dataset is not a Power Pool rule — the
    rules are eight operators and three fields — but an alpha that passes everything
    while drawing on a single dataset is additionally classed as ATOM, which is a second
    way for the same simulation to count. Crossing datasets stays legal, and is the whole
    point of the Pair lab.
    """
    same = [
        f
        for f in fields
        if f["dataset_id"] == first["dataset_id"] and f["field_id"] != first["field_id"]
    ]
    others = [f for f in fields if f["field_id"] != first["field_id"]]
    pool = same or others
    return rng.choice(pool) if pool else None


def mutate(rng: random.Random, tree: Node, fields: list[dict[str, Any]]) -> Node:
    """The same idea with exactly one thing about it changed.

    A product node is never the thing changed: it has nothing to vary but its two
    fields, and those are nodes of their own that this can pick instead.
    """
    where, node = rng.choice([p for p in paths(tree) if p[1].kind != "pair"])

    if node.kind == "field" and fields:
        changed = replace(node, op=str(rng.choice(fields)["field_id"]))
    elif node.kind == "ts":
        if rng.random() < 0.5:
            changed = replace(node, window=rng.choice(WINDOWS))
        else:
            changed = replace(node, op=rng.choice(TS_OPS))
    elif node.kind == "group":
        if rng.random() < 0.5:
            changed = replace(node, group=rng.choice(GROUPINGS))
        else:
            changed = replace(node, op=rng.choice(GROUP_OPS))
    elif node.op in CROSS_TWO:
        changed = replace(node, op=rng.choice(CROSS_TWO))
    else:
        changed = replace(node, op=rng.choice(CROSS_ONE))

    return swap(tree, where, changed)


def cross(rng: random.Random, left: Node, right: Node) -> Node | None:
    """Half of one idea joined to half of another.

    Only nodes of the same kind are exchanged, so the child is a legal tree by
    construction rather than by luck.
    """
    # The root is left out of the left-hand side: replacing it would exchange the whole
    # tree, which is not a crossover, it is the other parent.
    mine = [p for p in paths(left) if p[0]]
    theirs = paths(right)
    kinds = {node.kind for _, node in mine} & {node.kind for _, node in theirs}
    if not kinds:
        return None

    kind = rng.choice(sorted(kinds))
    where, _ = rng.choice([p for p in mine if p[1].kind == kind])
    _, graft = rng.choice([p for p in theirs if p[1].kind == kind])
    child = swap(left, where, graft)
    return child if child != left else None


# -- how good a finished alpha was -----------------------------------------

#: The piecewise utility from the research notes. Each part is worth at most one point,
#: so no single measure can carry an alpha on its own.
SHARPE_PLATEAU = 2.0
FITNESS_FLOOR, FITNESS_PLATEAU = 0.8, 1.5
TURNOVER_LOW, TURNOVER_HIGH = 0.05, 0.40


def utility(row: dict[str, Any]) -> float:
    """How much this finished alpha is worth breeding from.

    Sharpe is taken as an absolute value: a reliably negative alpha is a reliable alpha
    whose sign is the wrong way round, and the sign is free to change.
    """
    score = min(abs(float(row.get("sharpe") or 0.0)), SHARPE_PLATEAU) - 1.0

    fitness = min(abs(float(row.get("fitness") or 0.0)), FITNESS_PLATEAU)
    score += max(-1.0, (fitness - FITNESS_FLOOR) / (FITNESS_PLATEAU - FITNESS_FLOOR))

    turnover = float(row.get("turnover") or 0.0)
    if turnover < TURNOVER_LOW:
        score -= (TURNOVER_LOW - turnover) * 10
    elif turnover > TURNOVER_HIGH:
        score -= turnover - TURNOVER_HIGH

    return score - _failures(row.get("checks"))


def _failures(checks_json: Any) -> int:
    """One point off for every check the platform failed."""
    if not checks_json:
        return 0
    try:
        checks = json.loads(checks_json)
    except (json.JSONDecodeError, TypeError):
        return 0
    if not isinstance(checks, list):
        return 0
    return sum(
        1 for c in checks if isinstance(c, dict) and str(c.get("result", "")).upper() == "FAIL"
    )


# -- the lab ---------------------------------------------------------------


class Inventor:
    """Builds new expression shapes, and breeds the ones that worked."""

    def __init__(
        self,
        queries: CatalogQueries,
        harvester: SeedHarvester,
        vault: AlphaVault,
        skeletons: Any = None,
        auth: Any = None,
    ) -> None:
        self.queries = queries
        self.harvester = harvester
        self.vault = vault
        self.skeletons = skeletons
        #: Optional. Used to drop a shape whose operators this account cannot run.
        self.auth = auth

    async def plan(
        self,
        *,
        scope: Tuple4,
        target: int = 500,
        neutralization: str = "SUBINDUSTRY",
        decay: int = 4,
        truncation: float = 0.08,
        per_dataset: int = 2,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """The next generation, and the simulations that would run it. Costs nothing."""
        target = max(1, min(target, MAX_SIMULATIONS))

        fields = await self.harvester.best_fields(
            scope, per_dataset=per_dataset, min_coverage=MIN_COVERAGE, limit=80
        )
        if not fields:
            return _nothing(
                "There is no downloaded data in this market to build anything out of "
                "yet. Download it first."
            )
        # Sorted before anything is drawn from it. Coverage alone leaves ties — a hundred
        # fields can share it — and the database is free to return tied rows in any
        # order, which would make the same seed produce a different draw each time. A
        # seed nobody can reproduce is not a seed.
        fields.sort(key=lambda f: (-(f.get("coverage") or 0.0), str(f["field_id"])))

        known = await self._operators()
        parents = await self._parents(scope)
        saturated = await self._saturated(scope)

        from ..plan.day import today

        rng = random.Random(seed if seed is not None else f"{today()}-{scope.label}-invent")

        seen = {_normalise(p[1]) for p in parents}
        made: dict[str, str] = {}
        counts = {"crossover": 0, "mutation": 0, "fresh": 0}
        skipped = 0

        # Generous headroom: most draws are rejected — by the Power Pool budget, by an
        # operator this account lacks, by a shape this market has answered, or by being a
        # duplicate — and stopping at the first miss would empty the day.
        for _ in range(target * 12):
            if len(made) >= target:
                break
            tree, how = _draw(rng, parents, fields)
            if tree is None:
                continue
            expression = tree.render()
            key = _normalise(expression)
            if key in seen:
                continue
            seen.add(key)
            if not _runnable(expression, known):
                continue
            if _shape(expression) in saturated:
                skipped += 1
                continue
            made[expression] = how
            counts[how] += 1

        if not made:
            return _nothing(
                "Nothing new came out of this market that would fit the Power Pool "
                "rules. Download more data, or try a different market."
            )

        settings = SimulationSettings(
            instrumentType=scope.instrument_type,
            region=scope.region,
            delay=scope.delay,
            universe=scope.universe,
            neutralization=neutralization,
            decay=decay,
            truncation=truncation,
        )
        requests = [
            SimulationRequest(type="REGULAR", settings=settings, regular=expression)
            for expression in made
        ]

        log.info(
            "invent.planned",
            scope=scope.label,
            parents=len(parents),
            simulations=len(requests),
            **counts,
        )
        return {
            "requests": requests,
            "expressions": [
                {"expression": expression, "from": how}
                for expression, how in list(made.items())[:24]
            ],
            "possible": len(made),
            "parents": len(parents),
            "bestParent": round(parents[0][0], 3) if parents else None,
            "counts": counts,
            "workedOutSkipped": skipped,
            "settings": settings.model_dump(by_alias=True, exclude_none=True),
            "note": _note(len(parents), counts),
        }

    # -- what it breeds from ----------------------------------------------

    async def _parents(self, scope: Tuple4) -> list[tuple[float, str, Node]]:
        """Finished alphas in this market that this grammar can read, best first.

        Deliberately not restricted to this lab's own output. On day one there is none,
        and a generation drawn from nothing is a random search — whereas anything already
        in the vault that fits the grammar is a starting point the platform has already
        graded.
        """
        try:
            rows = await self.vault.alphas(
                region=scope.region,
                delay=scope.delay,
                universe=scope.universe,
                instrument_type=scope.instrument_type,
                limit=300,
            )
        except Exception:
            log.warning("invent.parents_unavailable", scope=scope.label, exc_info=True)
            return []

        scored: list[tuple[float, str, Node]] = []
        for row in rows:
            if row.get("sharpe") is None:
                continue
            tree = parse(row.get("expression"))
            if tree is None:
                continue
            scored.append((utility(row), str(row["expression"]), tree))
        scored.sort(key=lambda entry: -entry[0])
        return scored[:POPULATION]

    async def _saturated(self, scope: Tuple4) -> set[str]:
        if self.skeletons is None:
            return set()
        try:
            return set(await self.skeletons.saturated(scope))
        except Exception:
            log.warning("invent.skeletons_unavailable", scope=scope.label, exc_info=True)
            return set()

    async def _operators(self) -> set[str]:
        if self.auth is None:
            return set()
        try:
            cached = await self.auth.cached_operators()
        except Exception:
            return set()
        return {str(o.get("name")) for o in cached} if cached else set()


# -- helpers ---------------------------------------------------------------


def _draw(
    rng: random.Random,
    parents: list[tuple[float, str, Node]],
    fields: list[dict[str, Any]],
) -> tuple[Node | None, str]:
    """One candidate, and where it came from."""
    if len(parents) < 2:
        if parents and rng.random() < ELITE_MUTATION_SHARE + MUTATION_SHARE:
            return mutate(rng, parents[0][2], fields), "mutation"
        return grow(rng, fields), "fresh"

    roll = rng.random()
    if roll < CROSSOVER_SHARE:
        left, right = rng.sample(parents, 2)
        return cross(rng, left[2], right[2]), "crossover"
    if roll < CROSSOVER_SHARE + ELITE_MUTATION_SHARE:
        elite = rng.choice(parents[:ELITES])
        return mutate(rng, elite[2], fields), "mutation"
    if roll < CROSSOVER_SHARE + ELITE_MUTATION_SHARE + MUTATION_SHARE:
        return mutate(rng, rng.choice(parents)[2], fields), "mutation"
    return grow(rng, fields), "fresh"


def _runnable(expression: str, known: set[str]) -> bool:
    """Power Pool legal, and built only out of operators this account has."""
    if count_operators(expression) > POWER_POOL_MAX_OPERATORS:
        return False
    if len(unique_data_fields(expression)) > POWER_POOL_MAX_FIELDS:
        return False
    return not (known and set(operators_in(expression)) - known)


def _normalise(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def _shape(expression: str) -> str:
    from ..plan.skeletons import skeleton

    return skeleton(expression)


def _note(parents: int, counts: dict[str, int]) -> str:
    if not parents:
        return (
            "Nothing has finished in this market yet, so this first batch is drawn from "
            "the grammar rather than bred from anything. Run it, and the next batch "
            "builds on whatever worked."
        )
    return (
        f"Built out of the {parents} best alphas already finished here: "
        f"{counts['crossover']} by crossing two of them, {counts['mutation']} by changing "
        f"one thing about one of them, and {counts['fresh']} from nothing at all."
    )


def _nothing(note: str) -> dict[str, Any]:
    return {
        "requests": [],
        "expressions": [],
        "possible": 0,
        "parents": 0,
        "bestParent": None,
        "counts": {"crossover": 0, "mutation": 0, "fresh": 0},
        "workedOutSkipped": 0,
        "settings": {},
        "note": note,
    }
