"""API keys, sealed at rest and rotated by remaining budget.

The point of holding several keys is that free-tier quota is per account. Two keys is
two hundred requests a day on a Lite model instead of one hundred. Rotation therefore
picks by *headroom*, not round-robin: round-robin spreads load evenly, which is exactly
wrong when one key is nearly spent and another is untouched.

Keys are sealed with the same vault as the BRAIN password and never leave the backend.
What the UI receives is a masked hint — enough to tell two keys apart, useless to
anyone who intercepts it.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
from sqlalchemy import select

from ..db.models import ApiKey, utcnow
from ..db.sqlite import Database
from ..security.vault import Vault
from .budget import Headroom, Ledger, seconds_until_reset
from .registry import ModelInfo, ModelRegistry

log = structlog.get_logger(__name__)

#: Domain separation for the vault, matching the pattern used for the BRAIN credentials.
#: A blob sealed as an API key cannot be substituted in as a password, or the reverse.
KEY_CONTEXT = "google-api-key"


class LLMError(Exception):
    """Something wrong with the LLM setup, phrased for whoever has to fix it."""


class NoKeysError(LLMError):
    def __init__(self, provider: str | None = None) -> None:
        if provider and provider != "google":
            super().__init__(
                f"No {provider} key has been added yet. Open AI Integration to add one — "
                "every provider listed there is free and needs no card."
            )
            return
        super().__init__(
            "No assistant key has been added yet. Get a free Google AI Studio key at "
            "https://aistudio.google.com/apikey, or pick another free provider under AI "
            "Integration. Keys from more than one account add their budgets together."
        )


class BudgetExhaustedError(LLMError):
    """Every key is out of budget for this model."""

    def __init__(self, model: str, states: list[Headroom]) -> None:
        daily = [s for s in states if s.blocked_by == "requests_per_day"]
        if daily and len(daily) == len(states):
            hours = seconds_until_reset() / 3600
            message = (
                f"Every key has used its daily allowance of {model}. It resets in about "
                f"{hours:.0f} hours (midnight Pacific). Switch to a model with a larger "
                "daily budget, or add another key."
            )
        else:
            wait = min((s.retry_after for s in states), default=60.0)
            message = (
                f"Every key is at its per-minute limit for {model}. Try again in about "
                f"{wait:.0f} seconds."
            )
        super().__init__(message)
        self.model = model
        self.states = states
        self.retry_after = min((s.retry_after for s in states), default=60.0)
        self.daily = bool(daily) and len(daily) == len(states)


def fingerprint(key: str) -> str:
    """Stable identity for a key, so the same one is not added twice.

    A hash, not the key: the database should not hold a second recoverable copy of a
    secret next to the sealed one.
    """
    return hashlib.sha256(key.strip().encode()).hexdigest()


def hint(key: str) -> str:
    """Enough of a key to recognise it, not enough to use it."""
    clean = key.strip()
    if len(clean) <= 10:
        return "…" + clean[-3:]
    return f"{clean[:6]}…{clean[-4:]}"


def serialise(row: ApiKey, usage: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": row.id,
        "label": row.label,
        "provider": row.provider,
        "hint": row.hint,
        "enabled": row.enabled,
        "lastOkAt": row.last_ok_at.isoformat() if row.last_ok_at else None,
        "lastError": row.last_error,
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "usage": usage or [],
    }


class KeyStore:
    """Holds the keys and decides which one to use next."""

    def __init__(self, db: Database, vault: Vault, ledger: Ledger) -> None:
        self.db = db
        self.vault = vault
        self.ledger = ledger

    # -- storage ---------------------------------------------------------

    async def add(self, key: str, label: str | None = None, provider: str = "google") -> ApiKey:
        clean = key.strip()
        if not clean:
            raise LLMError("That key is empty.")
        print_ = fingerprint(clean)

        async with self.db.session() as session:
            query = select(ApiKey).where(ApiKey.fingerprint == print_)
            if user_id:
                query = query.where(ApiKey.user_id == user_id)
            existing = await session.scalar(query)
            if existing is not None:
                raise LLMError(
                    f"That key is already stored as {existing.label!r}. Adding it twice "
                    "would not increase your quota — quota is per account."
                )
            count_query = select(ApiKey.id)
            if user_id:
                count_query = count_query.where(ApiKey.user_id == user_id)
            count = len(list((await session.scalars(count_query)).all()))
            row = ApiKey(
                label=label or f"Key {count + 1}",
                provider=provider,
                key_sealed=self.vault.seal(clean, context=KEY_CONTEXT),
                hint=hint(clean),
                fingerprint=print_,
                user_id=user_id,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
        log.info("llm.key.added", label=row.label, hint=row.hint, provider=provider, user_id=user_id)
        return row

    async def list(self, user_id: str | None = None) -> list[ApiKey]:
        async with self.db.session() as session:
            query = select(ApiKey)
            if user_id:
                query = query.where(ApiKey.user_id == user_id)
            query = query.order_by(ApiKey.id)
            return list((await session.scalars(query)).all())

    async def get(self, key_id: int) -> ApiKey | None:
        async with self.db.session() as session:
            return await session.get(ApiKey, key_id)

    async def secret(self, key_id: int) -> str:
        async with self.db.session() as session:
            row = await session.get(ApiKey, key_id)
            if row is None:
                raise LLMError(f"No key {key_id}.")
            return self.vault.open(row.key_sealed, context=KEY_CONTEXT)

    async def set_enabled(self, key_id: int, enabled: bool) -> ApiKey:
        async with self.db.session() as session:
            row = await session.get(ApiKey, key_id)
            if row is None:
                raise LLMError(f"No key {key_id}.")
            row.enabled = enabled
            await session.commit()
            await session.refresh(row)
        return row

    async def remove(self, key_id: int) -> None:
        async with self.db.session() as session:
            row = await session.get(ApiKey, key_id)
            if row is None:
                raise LLMError(f"No key {key_id}.")
            await session.delete(row)
            await session.commit()
        log.info("llm.key.removed", key_id=key_id)

    async def mark(self, key_id: int, *, error: str | None = None) -> None:
        """Record the outcome of a call, so a dead key is visible rather than mysterious."""
        async with self.db.session() as session:
            row = await session.get(ApiKey, key_id)
            if row is None:
                return
            if error is None:
                row.last_ok_at = utcnow()
                row.last_error = None
            else:
                row.last_error = error[:500]
            await session.commit()

    # -- rotation --------------------------------------------------------

    async def choose(
        self,
        model: ModelInfo,
        *,
        estimated_tokens: int = 4_000,
        user_id: str | None = None,
    ) -> int:
        """The key with the most daily budget left for this model."""
        rows = [
            r
            for r in await self.list(user_id=user_id)
            if r.enabled and r.provider == model.provider
        ]
        if not rows:
            raise NoKeysError(model.provider)

        states: list[Headroom] = []
        for row in rows:
            states.append(
                await self.ledger.allows(row.id, model, estimated_tokens=estimated_tokens)
            )

        usable = [s for s in states if s.available]
        if not usable:
            raise BudgetExhaustedError(model.id, states)

        best = max(usable, key=lambda s: (s.daily_remaining, -s.key_id))
        return best.key_id

    async def status(
        self,
        registry: ModelRegistry,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Keys, their health, and what budget remains — the whole picture in one call."""
        rows = await self.list(user_id=user_id)
        usage = await self.ledger.usage([r.id for r in rows])
        by_key: dict[int, list[dict[str, Any]]] = {}
        for entry in usage:
            by_key.setdefault(int(entry["keyId"]), []).append(entry)

        configured_providers = {r.provider for r in rows}
        text_models = registry.all("text")
        budget: list[dict[str, Any]] = []
        for model in text_models:
            # Only models for providers the user has configured keys for are reported
            if model.provider not in configured_providers:
                continue
            usable = [r for r in rows if r.enabled and r.provider == model.provider]
            remaining = 0
            for row in usable:
                state = await self.ledger.headroom(row.id, model)
                remaining += state.daily_remaining
            budget.append(
                {
                    "model": model.id,
                    "label": model.label,
                    "provider": model.provider,
                    "perKeyPerDay": model.rpd,
                    "remainingToday": remaining,
                    "bulk": model.bulk,
                }
            )

        return {
            "keys": [serialise(r, by_key.get(r.id, [])) for r in rows],
            "enabled": sum(1 for r in rows if r.enabled),
            "budget": budget,
            "resetInSeconds": round(seconds_until_reset()),
            "quotaTimezone": "America/Los_Angeles",
        }
