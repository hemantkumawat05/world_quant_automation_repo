"""The base every response model shares, and the two shapes every caller repeats.

The frontend reads camelCase; Python writes snake_case. Declaring that mapping once
means a field added to a model reaches the wire on its own. The hand-written
``to_dict()`` methods this replaces had to be edited twice for every new field, and the
second edit is the one that gets forgotten.

Lives at the top of the package rather than under ``api/`` so domain modules can build
their own response models without importing the layer that serves them.
"""

from __future__ import annotations

import json
from collections.abc import Container
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class Out(BaseModel):
    """Serialises to camelCase, accepts either spelling on the way in."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )


def camel_dict(obj: Any, exclude: Container[str] = ()) -> dict[str, Any]:
    """A dataclass on the wire, in the same camelCase :class:`Out` produces.

    For the dataclasses that stay dataclasses. These are *working objects* that happen
    to be serialised at the end — a ``Task`` the registry mutates, an ``Answer`` that
    accumulates its own context, a ``Report`` holding validation machinery — not wire
    models with a method attached. Making them :class:`Out` would put validation on
    every assignment in a hot path to save one line each. Same reason as above, though:
    listing the fields a second time is a list that goes stale.

    ``exclude`` drops fields that are internal to the calculation rather than part of
    the answer. Anything computed rather than stored is not a field, so merge it in:
    ``camel_dict(self) | {"canMultiSimulate": self.can_multi_simulate}``.
    """
    return {to_camel(k): v for k, v in asdict(obj).items() if k not in exclude}


def loads_or(text: str, **default: Any) -> dict[str, Any]:
    """A model's JSON reply, or ``default`` when it is not usable.

    Every prompt that asks for JSON gets it *mostly*, and mostly is not a contract. A
    reply the consultant can read is worth more than a parse error, so the callers here
    degrade rather than raise — and they all degraded the same way, in four places.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default
    return payload if isinstance(payload, dict) else default


def clip(text: Any, limit: int) -> str:
    """One line, at most ``limit`` characters, broken on a word.

    Whitespace is collapsed first: descriptions arrive from the platform with newlines
    in them, and a prompt or a table cell wants one line.
    """
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[:limit].rsplit(" ", 1)[0] + "…"
