"""Calling the model.

Every call goes through :meth:`LLMService.generate`, which does the same four things in
the same order: pick a key with budget, send, record what it actually cost, and — when
Google disagrees with our accounting — mark that pair spent and try the next key rather
than retrying into the same wall.

Rotation is only worth having if it is *informed*. A retry loop that tries each key in
turn until one works will burn a request from every key on a bad day. Choosing by
remaining daily budget, and believing a ``429`` immediately, spends one.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from ..catalog.queries import CatalogQueries, Tuple4
from ..db.sqlite import Database
from ..security.vault import Vault
from .budget import Ledger
from .context import ContextBuilder, estimate_tokens
from .keys import BudgetExhaustedError, KeyStore, LLMError, NoKeysError
from .prompts import PROMPTS, PromptContext
from .registry import DEFAULT_MODEL, ModelInfo, ModelRegistry

log = structlog.get_logger(__name__)

#: How long a single generation may take before we give up on it.
TIMEOUT_SECONDS = 180.0


class UnknownPromptError(LLMError):
    """A prompt slug that does not ship with the application."""

    def __init__(self, ref: str) -> None:
        super().__init__(
            f"There is no prompt called {ref!r}. The ones that exist are: "
            + ", ".join(sorted(PROMPTS))
            + "."
        )
        self.ref = ref


@dataclass(slots=True)
class Answer:
    """One model response, with everything it cost."""

    text: str
    model: str
    key_id: int
    key_hint: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    #: What was put in front of the model, so the user can see it. "Hide nothing."
    context: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "keyId": self.key_id,
            "keyHint": self.key_hint,
            "usage": {
                "promptTokens": self.prompt_tokens,
                "outputTokens": self.output_tokens,
                "thinkingTokens": self.thinking_tokens,
                "totalTokens": self.total_tokens,
            },
            "context": self.context,
            "attempts": self.attempts,
        }


def _is_rate_limit(exc: Exception) -> tuple[bool, bool]:
    """``(rate limited, daily)`` — read from whatever the SDK gives us.

    The SDK's exception types have moved between releases, so this reads the message and
    any ``code``/``status`` attribute rather than catching a class that may be renamed.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).lower()
    limited = code == 429 or "429" in text or "resource_exhausted" in text or "quota" in text
    daily = "per day" in text or "perday" in text or "daily" in text or "requests per day" in text
    return limited, daily


def _plain_reason(exc: Exception) -> str | None:
    """The common failures, said in words the person reading them can act on.

    Google's errors arrive as a wall of JSON. Someone who has been using this for ten
    minutes and pasted a key with a character missing should be told that, not shown
    ``INVALID_ARGUMENT`` and a list of ``@type`` URLs. The raw text is still kept against
    the key as ``lastError`` for anyone who wants it.
    """
    text = str(exc).lower()
    if "api_key_invalid" in text or "api key not valid" in text:
        return (
            "That Google AI Studio key was not accepted. Check it was copied whole, and "
            "that it has not been deleted at aistudio.google.com."
        )
    if "permission_denied" in text or "403" in text:
        return (
            "Google refused that key. It may not have access to this model, or the "
            "project it belongs to may have been closed."
        )
    if "not found" in text and "model" in text:
        return "That model is no longer available from Google. Pick a different one."
    return None


class LLMService:
    """The assistant: keys, budget, context and prompts in one place."""

    def __init__(
        self,
        db: Database,
        vault: Vault,
        queries: CatalogQueries,
        registry: ModelRegistry,
    ) -> None:
        self.db = db
        self.registry = registry
        self.ledger = Ledger(db)
        self.keys = KeyStore(db, vault, self.ledger)
        self.context = ContextBuilder(queries)
        self._clients: dict[int, Any] = {}

    # -- plumbing --------------------------------------------------------

    def _client(self, key_id: int, secret: str, provider: str = "google") -> Any:
        """The client for one key. Google speaks its own protocol; everything else
        speaks chat-completions, which one small client covers."""
        client = self._clients.get(key_id)
        if client is None:
            if provider == "google":
                from google import genai

                client = genai.Client(api_key=secret)
            else:
                from .openai_compat import OpenAICompatible
                from .providers import get as provider_spec

                spec = provider_spec(provider)
                if not spec.base_url:
                    raise LLMError(f"{provider!r} is not a provider this application knows.")
                client = OpenAICompatible(spec.base_url, secret, timeout=TIMEOUT_SECONDS)
            self._clients[key_id] = client
        return client

    def forget(self, key_id: int) -> None:
        self._clients.pop(key_id, None)

    def model_for(self, model_id: str | None) -> ModelInfo:
        return self.registry.require(model_id or DEFAULT_MODEL)

    async def require_keys(self) -> None:
        """Fail on the missing key before anything else.

        Checked ahead of the catalog because the two blockers are not equal: without a
        key nothing here works at all, whereas an unsynced scope is a single sync away.
        Checking the catalog first would also mean that syncing data *changes* the error
        message, which reads as though the sync broke something.
        """
        if not any(row.enabled for row in await self.keys.list()):
            raise NoKeysError()

    # -- the one call ----------------------------------------------------

    async def generate(
        self,
        *,
        system: str,
        user: str,
        model_id: str | None = None,
        response_schema: Any = None,
        temperature: float = 0.7,
        thinking: str | None = None,
    ) -> Answer:
        """Send one prompt, rotating keys by remaining budget.

        ``thinking`` is one of Google's thinking levels. It is not a free upgrade:
        thinking tokens are billed against the same per-minute budget as the answer, so
        asking the model to think harder costs the consultant real allowance.
        """
        from google.genai import types

        model = self.model_for(model_id)
        estimate = estimate_tokens(system) + estimate_tokens(user)
        attempts: list[dict[str, Any]] = []
        tried: set[int] = set()

        thinking_config = types.ThinkingConfig(thinking_level=thinking) if thinking else None
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            **({"thinking_config": thinking_config} if thinking_config else {}),
            **({"response_mime_type": "application/json"} if response_schema else {}),
            **({"response_schema": response_schema} if response_schema else {}),
        )

        while True:
            try:
                key_id = await self.keys.choose(model, estimated_tokens=estimate)
            except BudgetExhaustedError as exc:
                if attempts:
                    # Some key worked earlier in this loop's life; report both facts.
                    exc.args = (f"{exc.args[0]} Already tried: {len(attempts)} key(s).",)
                raise

            if key_id in tried:
                # choose() keeps returning a key we already failed on, which means our
                # accounting and Google's disagree in a way one more call will not fix.
                raise LLMError(
                    "Every key with budget left has just been rejected by Google. Its "
                    "limits may have changed. Try again in a minute, or add another key."
                )
            tried.add(key_id)

            row = await self.keys.get(key_id)
            secret = await self.keys.secret(key_id)
            provider = str(getattr(row, "provider", "google") or "google")
            client = self._client(key_id, secret, provider)

            try:
                call = (
                    client.aio.models.generate_content(model=model.id, contents=user, config=config)
                    if provider == "google"
                    else client.generate(
                        model=model.id,
                        system=system,
                        user=user,
                        temperature=temperature,
                        json_mode=response_schema is not None,
                    )
                )
                response = await asyncio.wait_for(call, timeout=TIMEOUT_SECONDS)
            except TimeoutError as exc:
                await self.keys.mark(key_id, error="Timed out")
                raise LLMError(
                    f"{model.label} did not answer within {TIMEOUT_SECONDS:.0f} seconds. "
                    "Try a smaller question or a lighter model."
                ) from exc
            except Exception as exc:
                limited, daily = _is_rate_limit(exc)
                attempts.append({"keyId": key_id, "error": str(exc)[:300], "rateLimited": limited})
                await self.keys.mark(key_id, error=str(exc)[:300])
                if limited:
                    # Google is the authority. Mark this pair spent and rotate rather
                    # than retrying into a limit we evidently mis-tracked.
                    await self.ledger.penalise(key_id, model, daily=daily)
                    self.forget(key_id)
                    log.warning("llm.rate_limited", key_id=key_id, model=model.id, daily=daily)
                    continue
                plain = _plain_reason(exc)
                raise LLMError(plain or f"{model.label} could not be reached: {exc}") from exc

            usage = getattr(response, "usage_metadata", None)
            total = int(getattr(usage, "total_token_count", 0) or 0) or estimate
            await self.ledger.record(key_id, model.id, total)
            await self.keys.mark(key_id)

            return Answer(
                text=(response.text or "").strip(),
                model=model.id,
                key_id=key_id,
                key_hint=row.hint if row else "",
                prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                thinking_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
                total_tokens=total,
                attempts=attempts,
            )

    # -- health ----------------------------------------------------------

    async def check_key(self, key_id: int) -> dict[str, Any]:
        """Confirm a key works, without spending a generation request.

        ``models.list`` is not billed against the generation quotas, so a health check
        costs nothing that matters.
        """
        row = await self.keys.get(key_id)
        provider = str(getattr(row, "provider", "google") or "google")
        secret = await self.keys.secret(key_id)
        client = self._client(key_id, secret, provider)
        try:
            if provider == "google":
                names = [m.name or "" async for m in await client.aio.models.list()]
            else:
                names = await client.models()
        except Exception as exc:
            await self.keys.mark(key_id, error=str(exc)[:300])
            return {"keyId": key_id, "ok": False, "error": str(exc)[:300]}

        await self.keys.mark(key_id)
        added = self.registry.merge_discovered(names, provider)
        return {"keyId": key_id, "ok": True, "models": len(names), "newModels": added}

    async def check_all(self) -> list[dict[str, Any]]:
        return [await self.check_key(row.id) for row in await self.keys.list()]

    # -- features --------------------------------------------------------

    async def run(
        self,
        prompt_ref: str,
        user_input: str,
        *,
        scope: Tuple4 | None = None,
        dataset_ids: list[str] | None = None,
        model_id: str | None = None,
    ) -> Answer:
        """Run a shipped prompt, assembling whatever context it declares.

        The one path every prompt goes through: the prompt names the data it wants and
        this assembles it, so no feature carries its own idea of what its prompt needs.
        """
        await self.require_keys()
        prompt = PROMPTS.get(prompt_ref)
        if prompt is None:
            raise UnknownPromptError(prompt_ref)
        assembled, meta = await self.assemble(prompt.context, scope, dataset_ids, user_input)
        answer = await self.generate(
            system=prompt.body,
            user=assembled,
            model_id=model_id or prompt.model,
            temperature=prompt.temperature,
        )
        answer.context = {"prompt": prompt.slug, **meta}
        return answer

    async def assemble(
        self,
        context: PromptContext,
        scope: Tuple4 | None,
        dataset_ids: list[str] | None,
        user_input: str,
    ) -> tuple[str, dict[str, Any]]:
        """Build the user turn for a prompt's declared context."""
        if context is PromptContext.NONE:
            return user_input, {"kind": "none"}

        if scope is None:
            raise LLMError(
                "This prompt needs a data scope — instrument type, region, delay and "
                "universe — because it is given catalog data to reason over."
            )

        if context is PromptContext.DATASET_FIELDS and dataset_ids:
            fields, meta = await self.context.fields_for(scope, dataset_ids)
            return f"{fields}\n\n---\n\n{user_input}", {"kind": "dataset_fields", **meta}

        tree, meta = await self.context.render(scope)
        if meta["counts"]["datasets"] == 0:
            raise LLMError(
                f"Nothing is synced for {scope.label}, so there is no catalog to reason "
                "over. Sync this scope in the Data Explorer first."
            )
        return (
            f"{tree}\n\n---\n\nTHE USER'S IDEA OR QUESTION:\n{user_input}",
            {"kind": "catalog_tree", **meta},
        )

    async def advise(self, question: str, scope: Tuple4, *, model_id: str | None = None) -> Answer:
        """Which datasets could implement this idea."""
        return await self.run("data_advisor", question, scope=scope, model_id=model_id)

    async def power_pool_batch(
        self,
        scope: Tuple4,
        *,
        model_id: str | None = None,
        dataset_ids: list[str] | None = None,
        count: int = 30,
        operators: list[str] | None = None,
        brief: str = "",
        seeds: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate Power Pool expressions and check every one before it costs anything.

        The model is asked for finished expressions rather than templates, and none of
        them is trusted: each is checked against the platform's operator list, the fields
        actually available in this scope, and the Power Pool limits. What comes back is
        the survivors and — just as usefully — why the rest were rejected.
        """
        await self.require_keys()
        parts: list[str] = []

        if operators:
            parts.append("AVAILABLE OPERATORS (use only these):\n" + ", ".join(sorted(operators)))

        fields, meta = await self.context.fields_for(scope, dataset_ids or [])
        if not fields.strip():
            tree, meta = await self.context.render(scope)
            parts.append(tree)
            parts.append(
                "No datasets were chosen, so pick fields you can see above and name them "
                "exactly. If you need a field that is not listed, say so instead of "
                "inventing one."
            )
        else:
            parts.append("DATA FIELDS AVAILABLE IN THIS SCOPE:\n" + fields)

        if seeds:
            parts.append(
                "ALPHAS THAT ALREADY WORK HERE — build on these ideas rather than "
                "repeating them:\n" + "\n".join(f"  {e}" for e in seeds[:20])
            )
        if brief:
            parts.append(f"WHAT THE USER IS LOOKING FOR:\n{brief}")
        parts.append(f"SCOPE: {scope.label}\n\nWrite {count} candidate expressions.")

        prompt = PROMPTS["power_pool_author"]
        answer = await self.generate(
            system=prompt.body,
            user="\n\n---\n\n".join(parts),
            model_id=model_id or prompt.model,
            temperature=prompt.temperature,
            response_schema={
                "type": "object",
                "properties": {
                    "alphas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                                "idea": {"type": "string"},
                                "fields": {"type": "array", "items": {"type": "string"}},
                                "operators": {"type": "integer"},
                            },
                            "required": ["expression", "idea"],
                        },
                    }
                },
                "required": ["alphas"],
            },
        )
        answer.context = {"kind": "power_pool", "scope": scope.label, **meta}
        return {"answer": answer, "proposed": _parse_alphas(answer.text)}

    async def explain(self, subject: str, *, model_id: str | None = None) -> Answer:
        return await self.run("explainer", subject, model_id=model_id)


def _parse_alphas(text: str) -> list[dict[str, Any]]:
    """Pull the candidate list out of a response.

    The schema asks for JSON and models mostly comply, but "mostly" is not a contract
    and a batch that cannot be parsed is a whole request wasted. A fenced block is tried
    next, and an empty list is returned rather than raising.
    """
    for candidate in (text, *(m.group(1) for m in FENCE.finditer(text))):
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("alphas"), list):
            return [a for a in payload["alphas"] if isinstance(a, dict) and a.get("expression")]
        if isinstance(payload, list):
            return [a for a in payload if isinstance(a, dict) and a.get("expression")]
    return []


FENCE = re.compile(r"```(?:ya?ml|json)?\s*\n(.*?)```", re.DOTALL)


__all__ = [
    "Answer",
    "BudgetExhaustedError",
    "LLMError",
    "LLMService",
    "NoKeysError",
    "UnknownPromptError",
]
