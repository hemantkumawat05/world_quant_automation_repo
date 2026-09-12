"""What the model is told about the data.

The payload is the organisational hierarchy — nine categories, their subcategories, and
every dataset with its metadata — and deliberately **not the individual data fields**.

That is the whole trick. A synced scope holds tens of thousands of fields; serialising
them would fill the context window with names like ``fnd6_newqv1300_ivltq`` and leave no
room for the model to think. The hierarchy is a few hundred rows, fits comfortably, and
is enough to answer the question actually being asked — *which dataset would implement
this idea* — because a dataset's name, description and coverage say what it contains.
Fields are fetched on demand once the model has narrowed to a dataset.

Two numbers on each dataset are worth more than they look:

* **value score** is BRAIN's own estimate of how much signal a dataset still holds.
* **alpha count** is how many alphas already use it. A dataset with a high value score
  and a low alpha count is where an edge is most likely to survive, and saying so in the
  payload is more useful than hoping the model infers it.

Everything is rendered compactly, because tokens spent on JSON punctuation are tokens
not spent on the answer.
"""

from __future__ import annotations

from typing import Any

import structlog

from ..catalog.queries import CatalogQueries, Tuple4
from ..schemas import clip

log = structlog.get_logger(__name__)

#: Roughly four characters per token. Only used to warn before a call, never to bill.
CHARS_PER_TOKEN = 4

#: Datasets per subcategory in the compact rendering. High enough to include everything
#: in practice; a guard rather than a policy.
MAX_DATASETS = 400

#: Descriptions are trimmed. The first sentence carries what a dataset is; the rest is
#: usually vendor boilerplate.
DESCRIPTION_CHARS = 180


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}"


class ContextBuilder:
    """Turns the synced catalog into something a model can reason over."""

    def __init__(self, queries: CatalogQueries) -> None:
        self.queries = queries

    async def tree(self, scope: Tuple4) -> dict[str, Any]:
        """The category → subcategory → dataset hierarchy, with metadata, without fields."""
        counts = await self.queries.counts(scope)
        categories = await self.queries.category_tree(scope)
        datasets = await self.queries.datasets(scope)

        by_subcategory: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in datasets[:MAX_DATASETS]:
            key = (str(row.get("category_id") or ""), str(row.get("subcategory_id") or ""))
            by_subcategory.setdefault(key, []).append(row)

        nodes: list[dict[str, Any]] = []
        for category in categories:
            subcategories = []
            for sub in category["subcategories"]:
                rows = by_subcategory.get((category["id"], sub["id"]), [])
                subcategories.append(
                    {
                        "id": sub["id"],
                        "name": sub["name"],
                        "fields": sub["fields"],
                        "datasets": [
                            {
                                "id": r["dataset_id"],
                                "name": r.get("name") or r["dataset_id"],
                                "description": clip(r.get("description"), DESCRIPTION_CHARS),
                                "fields": r.get("field_count"),
                                "coverage": r.get("coverage"),
                                "valueScore": r.get("value_score"),
                                "userCount": r.get("user_count"),
                                "alphaCount": r.get("alpha_count"),
                                "pyramidMultiplier": r.get("pyramid_multiplier"),
                            }
                            for r in rows
                        ],
                    }
                )
            nodes.append(
                {
                    "id": category["id"],
                    "name": category["name"],
                    "fields": category["fields"],
                    "datasets": category["datasets"],
                    "subcategories": subcategories,
                }
            )

        return {"scope": scope.label, "counts": counts, "categories": nodes}

    async def render(self, scope: Tuple4) -> tuple[str, dict[str, Any]]:
        """The tree as compact text, plus what it cost.

        Text rather than JSON: the same information in half the tokens, and models read
        an indented outline at least as well as they read braces.
        """
        tree = await self.tree(scope)
        counts = tree["counts"]

        lines = [
            f"DATA CATALOG — {tree['scope']}",
            (
                f"{counts['categories']} categories · {counts['subcategories']} subcategories "
                f"· {counts['datasets']} datasets · {counts['fields']} fields"
            ),
            "",
            "Columns per dataset: id | name | fields | coverage | value score | alphas built",
            "A high value score with a low alpha count is an under-explored dataset.",
            "",
        ]

        for category in tree["categories"]:
            lines.append(
                f"## {category['name']} [{category['id']}] "
                f"— {category['datasets']} datasets, {category['fields']} fields"
            )
            for sub in category["subcategories"]:
                if not sub["datasets"]:
                    continue
                lines.append(f"  ### {sub['name']} [{sub['id']}]")
                for dataset in sub["datasets"]:
                    lines.append(
                        f"    - {dataset['id']} | {dataset['name']} | "
                        f"{_number(dataset['fields'])}f | "
                        f"cov {_number(dataset['coverage'])} | "
                        f"vs {_number(dataset['valueScore'])} | "
                        f"{_number(dataset['alphaCount'])} alphas"
                    )
                    if dataset["description"]:
                        lines.append(f"        {dataset['description']}")
            lines.append("")

        text = "\n".join(lines)
        meta = {
            "scope": tree["scope"],
            "counts": counts,
            "characters": len(text),
            "estimatedTokens": estimate_tokens(text),
        }
        log.info("llm.context.built", **meta)
        return text, meta

    async def fields_for(
        self, scope: Tuple4, dataset_ids: list[str], limit: int = 60
    ) -> tuple[str, dict[str, Any]]:
        """The fields of specific datasets, once the model has narrowed to them.

        The second half of the bargain: keeping fields out of the first payload is only
        reasonable if they can be asked for afterwards.
        """
        from ..catalog.queries import FieldFilter

        blocks: list[str] = []
        total = 0
        for dataset_id in dataset_ids[:12]:
            page = await self.queries.fields(
                scope,
                FieldFilter(dataset_ids=[dataset_id], sort_by="coverage", limit=limit),
            )
            total += int(page["total"])
            blocks.append(f"## {dataset_id} — {page['total']} fields")
            for row in page["results"]:
                blocks.append(
                    f"  - {row['field_id']} ({row.get('field_type')}, "
                    f"cov {_number(row.get('coverage'))}, "
                    f"{_number(row.get('alpha_count'))} alphas): "
                    f"{clip(row.get('description'), DESCRIPTION_CHARS)}"
                )
            blocks.append("")

        text = "\n".join(blocks)
        return text, {
            "datasets": dataset_ids[:12],
            "fields": total,
            "estimatedTokens": estimate_tokens(text),
        }
