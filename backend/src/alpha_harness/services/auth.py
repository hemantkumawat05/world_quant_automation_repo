"""Credential storage and session lifecycle.

Ties three things together: the vault (sealing secrets), SQLite (persisting them), and
the BRAIN authenticator (using them). Supports multi-tenant session pools and JWT generation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select

from ..brain.auth import Authenticator, SessionInfo
from ..brain.client import BrainClient
from ..brain.endpoints import BrainEndpoints
from ..db.models import BrainSessionRow, Credential, MetadataCache, utcnow
from ..db.sqlite import Database
from ..security.jwt import create_access_token
from ..security.vault import Vault

log = structlog.get_logger(__name__)

PASSWORD_CONTEXT = "brain-password"
COOKIE_CONTEXT = "brain-cookies"


class NoCredentialError(RuntimeError):
    """No BRAIN credential has been stored and none was supplied."""


class AuthService:
    """The app's view of "are we signed in to BRAIN?"."""

    def __init__(
        self,
        db: Database,
        vault: Vault,
        endpoints: BrainEndpoints,
        *,
        authenticator: Authenticator | None = None,
    ) -> None:
        self.db = db
        self.vault = vault
        self.endpoints = endpoints
        self.auth = authenticator or Authenticator(endpoints)
        self._session = SessionInfo.anonymous()
        self._user_profile: dict[str, Any] | None = None

        # Multi-user session and endpoint caches
        self._user_sessions: dict[str, SessionInfo] = {}
        self._user_endpoints: dict[str, BrainEndpoints] = {}
        self._user_profiles: dict[str, dict[str, Any]] = {}
        self._pending_verifications: dict[str, str] = {}

    def _create_endpoints(self, cookies: list[dict[str, Any]] | None = None) -> BrainEndpoints:
        base_client = self.endpoints.client
        client = BrainClient(
            base_client.base_url,
            min_retry_after=base_client.min_retry_after,
            poll_timeout=base_client.poll_timeout,
            min_request_interval=base_client.throttle.min_interval,
            default_attempts=base_client.default_attempts,
        )
        if cookies:
            client.load_cookies(cookies)
        return BrainEndpoints(client)

    @property
    def session(self) -> SessionInfo:
        """Last known global session state (for backwards compatibility)."""
        return self._session

    async def get_user_profile(self, user_id: str | None = None) -> dict[str, Any]:
        """Cached user profile from BRAIN /users/{userId}."""
        if not user_id:
            if not self._session.authenticated or not self._session.user_id:
                return {}
            user_id = self._session.user_id

        if user_id in self._user_profiles:
            return self._user_profiles[user_id]

        endpoints = self._user_endpoints.get(user_id, self.endpoints)
        try:
            profile = await endpoints.get_user(user_id)
            if profile:
                self._user_profiles[user_id] = profile
                first = str(profile.get("firstName") or "").strip()
                last = str(profile.get("lastName") or "").strip()
                full_name = (
                    profile.get("fullName")
                    or profile.get("name")
                    or (f"{first} {last}".strip() or None)
                )
                if user_id in self._user_sessions and full_name:
                    self._user_sessions[user_id].full_name = full_name
                if self._session.user_id == user_id and full_name:
                    self._session.full_name = full_name
            return profile
        except Exception as exc:
            log.warning("brain.user_profile.failed", user_id=user_id, error=str(exc))
            return {}

    async def get_user_context(self, user_id: str) -> tuple[SessionInfo, BrainEndpoints]:
        """Retrieve or restore session and endpoints for a specific user."""
        if user_id in self._user_sessions and user_id in self._user_endpoints:
            return self._user_sessions[user_id], self._user_endpoints[user_id]

        # Try to restore from database
        cookies = await self._load_cookies(user_id=user_id)
        if cookies:
            endpoints = self._create_endpoints(cookies)
            authenticator = Authenticator(endpoints)
            restored = await authenticator.restore(cookies)
            if restored and restored.authenticated:
                self._user_sessions[user_id] = restored
                self._user_endpoints[user_id] = endpoints
                await self.get_user_profile(user_id)
                return restored, endpoints

        # Try re-login with stored credentials if available
        cred = await self.get_credential(user_id=user_id)
        if cred:
            email, password = cred
            endpoints = self._create_endpoints()
            authenticator = Authenticator(endpoints)
            info = await authenticator.login(email, password)
            if info and info.authenticated:
                self._user_sessions[user_id] = info
                self._user_endpoints[user_id] = endpoints
                await self._save_cookies(info, endpoints=endpoints, user_id=user_id)
                await self.get_user_profile(user_id)
                return info, endpoints

        return SessionInfo.anonymous(), self.endpoints

    # -- credential storage ----------------------------------------------

    async def store_credential(self, email: str, password: str, user_id: str | None = None) -> None:
        """Save (or replace) the BRAIN login, sealed at rest."""
        sealed = self.vault.seal(password, context=PASSWORD_CONTEXT)
        async with self.db.session() as session:
            existing = (
                (await session.execute(select(Credential).where(Credential.email == email)))
                .scalars()
                .first()
            )
            if existing is not None:
                existing.password_sealed = sealed
                if user_id:
                    existing.user_id = user_id
            else:
                session.add(Credential(email=email, password_sealed=sealed, user_id=user_id))
        log.info("credential.stored", email=_mask(email), user_id=user_id)

    async def get_credential(self, user_id: str | None = None) -> tuple[str, str] | None:
        """Return the stored ``(email, password)``, unsealed."""
        async with self.db.session() as session:
            if user_id:
                query = select(Credential).where(Credential.user_id == user_id)
            else:
                query = select(Credential).order_by(
                    Credential.last_login_at.desc().nulls_last(), Credential.id.desc()
                )
            credential = (await session.execute(query.limit(1))).scalars().first()
            if credential is None:
                return None
            password = self.vault.open(credential.password_sealed, context=PASSWORD_CONTEXT)
            return credential.email, password

    async def stored_email(self, user_id: str | None = None) -> str | None:
        async with self.db.session() as session:
            if user_id:
                query = select(Credential).where(Credential.user_id == user_id)
            else:
                query = select(Credential).order_by(
                    Credential.last_login_at.desc().nulls_last(), Credential.id.desc()
                )
            credential = (await session.execute(query.limit(1))).scalars().first()
            return credential.email if credential else None

    async def forget(self, user_id: str | None = None) -> None:
        """Remove the credential and any cached session."""
        async with self.db.session() as session:
            if user_id:
                for cred in (
                    await session.execute(select(Credential).where(Credential.user_id == user_id))
                ).scalars():
                    await session.delete(cred)
            else:
                for cred in (await session.execute(select(Credential))).scalars():
                    await session.delete(cred)
        if user_id:
            self._user_sessions.pop(user_id, None)
            self._user_endpoints.pop(user_id, None)
            self._user_profiles.pop(user_id, None)
            await self._clear_cookies(user_id=user_id)
        else:
            self.endpoints.client.clear_cookies()
            self._session = SessionInfo.anonymous()
            self._user_sessions.clear()
            self._user_endpoints.clear()
            self._user_profiles.clear()
            await self._clear_cookies()

    # -- session ---------------------------------------------------------

    async def restore(self, user_id: str | None = None) -> SessionInfo:
        """Reuse a cached cookie jar if it is still valid."""
        cookies = await self._load_cookies(user_id=user_id)
        if not cookies:
            return SessionInfo.anonymous()

        endpoints = self._create_endpoints(cookies)
        authenticator = Authenticator(endpoints)
        restored = await authenticator.restore(cookies)

        if restored is None or not restored.authenticated:
            session_info = restored or SessionInfo.anonymous()
            if user_id:
                self._user_sessions[user_id] = session_info
            else:
                self._session = session_info
            return session_info

        actual_user_id = restored.user_id or user_id or "default"
        self._user_sessions[actual_user_id] = restored
        self._user_endpoints[actual_user_id] = endpoints
        self._session = restored

        await self._save_cookies(restored, endpoints=endpoints, user_id=actual_user_id)
        await self._warm_operators(endpoints=endpoints)
        return restored

    async def login(
        self, email: str | None = None, password: str | None = None
    ) -> tuple[SessionInfo, str | None]:
        """Sign in, storing the credential and generating an access token."""
        supplied = bool(email and password)
        if not supplied:
            credential = await self.get_credential()
            if credential is None:
                raise NoCredentialError(
                    "No BRAIN credentials stored. Sign in with your BRAIN email and password."
                )
            email, password = credential
        assert email is not None and password is not None

        endpoints = self._create_endpoints()
        authenticator = Authenticator(endpoints)

        norm_email = (email or "").strip().lower()
        pending = self._pending_verifications.get(norm_email) or self._session.verification_url
        info: SessionInfo | None = None
        if pending:
            info = await authenticator.verify(pending, email, password)

        if info is None or not info.authenticated:
            fresh_info = await authenticator.login(email, password)
            if fresh_info.authenticated or fresh_info.verification_url or not info:
                info = fresh_info

        if info.verification_url:
            self._pending_verifications[norm_email] = info.verification_url
            self._session = info
        else:
            self._pending_verifications.pop(norm_email, None)

        token: str | None = None
        if info.authenticated and info.user_id:
            token = create_access_token(user_id=info.user_id, email=email, key=self.vault.key)
            self._user_sessions[info.user_id] = info
            self._user_endpoints[info.user_id] = endpoints
            self._session = info

            if supplied:
                await self.store_credential(email, password, user_id=info.user_id)
            await self._touch_last_login(email, user_id=info.user_id)
            await self._save_cookies(info, endpoints=endpoints, user_id=info.user_id)
            await self.get_user_profile(info.user_id)
            await self._warm_operators(endpoints=endpoints)

        return info, token

    async def status(self, user_id: str | None = None, *, refresh: bool = False) -> SessionInfo:
        """Current state for a specific user."""
        if not user_id:
            return self._session

        if not refresh and user_id in self._user_sessions:
            return self._user_sessions[user_id]

        info, _ = await self.get_user_context(user_id)
        if refresh and info.authenticated and user_id in self._user_endpoints:
            fresh = await Authenticator(self._user_endpoints[user_id]).status()
            if fresh.authenticated and not fresh.full_name and info.full_name:
                fresh.full_name = info.full_name
            self._user_sessions[user_id] = fresh
            info = fresh
        return info

    async def logout(self, user_id: str | None = None) -> None:
        if user_id:
            endpoints = self._user_endpoints.get(user_id)
            if endpoints:
                with contextlib.suppress(Exception):
                    await Authenticator(endpoints).logout()
            self._user_sessions.pop(user_id, None)
            self._user_endpoints.pop(user_id, None)
            self._user_profiles.pop(user_id, None)
            await self._clear_cookies(user_id=user_id)
        else:
            self._user_profile = None
            await self.auth.logout()
            await self._clear_cookies()
            self._session = SessionInfo.anonymous()

    # -- cookie persistence ----------------------------------------------

    async def _load_cookies(self, user_id: str | None = None) -> list[dict[str, Any]] | None:
        async with self.db.session() as session:
            query = select(BrainSessionRow)
            if user_id:
                query = query.where(BrainSessionRow.user_id == user_id)
            query = query.order_by(BrainSessionRow.updated_at.desc()).limit(1)

            row = (await session.execute(query)).scalars().first()
            if row is None:
                return None
            try:
                return json.loads(self.vault.open(row.cookies_sealed, context=COOKIE_CONTEXT))
            except Exception:
                log.warning("session.cookies_unreadable", user_id=user_id)
                return None

    async def _save_cookies(
        self,
        info: SessionInfo,
        endpoints: BrainEndpoints | None = None,
        user_id: str | None = None,
    ) -> None:
        client = endpoints.client if endpoints else self.endpoints.client
        cookies = client.export_cookies()
        if not cookies:
            return
        sealed = self.vault.seal(json.dumps(cookies), context=COOKIE_CONTEXT)
        expires = datetime.fromtimestamp(info.expires_at, tz=UTC) if info.expires_at else None
        target_user = user_id or info.user_id

        async with self.db.session() as session:
            row = None
            if target_user:
                row = (
                    (
                        await session.execute(
                            select(BrainSessionRow).where(BrainSessionRow.user_id == target_user)
                        )
                    )
                    .scalars()
                    .first()
                )
            if row is None:
                # Find credential id
                cred = None
                if target_user:
                    cred = (
                        (
                            await session.execute(
                                select(Credential).where(Credential.user_id == target_user)
                            )
                        )
                        .scalars()
                        .first()
                    )
                cred_id = cred.id if cred else 1
                session.add(
                    BrainSessionRow(
                        credential_id=cred_id,
                        cookies_sealed=sealed,
                        user_id=target_user,
                        permissions=info.permissions,
                        expires_at=expires,
                    )
                )
            else:
                row.cookies_sealed = sealed
                row.user_id = target_user
                row.permissions = info.permissions
                row.expires_at = expires

    async def _clear_cookies(self, user_id: str | None = None) -> None:
        async with self.db.session() as session:
            query = select(BrainSessionRow)
            if user_id:
                query = query.where(BrainSessionRow.user_id == user_id)
            for row in (await session.execute(query)).scalars():
                await session.delete(row)

    async def _touch_last_login(self, email: str, user_id: str | None = None) -> None:
        async with self.db.session() as session:
            credential = (
                (await session.execute(select(Credential).where(Credential.email == email)))
                .scalars()
                .first()
            )
            if credential is not None:
                credential.last_login_at = utcnow()
                if user_id:
                    credential.user_id = user_id

    # -- platform metadata -----------------------------------------------

    async def refresh_metadata(self) -> dict[str, Any]:
        schema = await self.endpoints.settings_schema()
        await self._cache("settings_schema", schema)
        return schema

    async def cached_settings_schema(self) -> dict[str, Any] | None:
        return await self._read_cache("settings_schema")

    async def refresh_operators(self, endpoints: BrainEndpoints | None = None) -> list[dict[str, Any]]:
        target = endpoints or self.endpoints
        operators = await target.list_operators()
        payload = [o.model_dump(by_alias=True) for o in operators]
        await self._cache("operators", {"items": payload})
        return payload

    async def _warm_operators(self, endpoints: BrainEndpoints | None = None) -> None:
        try:
            await self.refresh_operators(endpoints)
        except Exception:
            log.warning("operators.refresh_failed", exc_info=True)

    async def cached_operators(self) -> list[dict[str, Any]] | None:
        cached = await self._read_cache("operators")
        return cached.get("items") if cached else None

    async def _cache(self, key: str, value: dict[str, Any]) -> None:
        async with self.db.session() as session:
            row = await session.get(MetadataCache, key)
            if row is None:
                session.add(MetadataCache(key=key, value=value))
            else:
                row.value = value
                row.fetched_at = utcnow()

    async def _read_cache(self, key: str) -> dict[str, Any] | None:
        async with self.db.session() as session:
            row = await session.get(MetadataCache, key)
            return row.value if row else None


async def _current_credential(session: Any) -> Credential | None:
    return (
        (
            await session.execute(
                select(Credential)
                .order_by(Credential.last_login_at.desc().nulls_last(), Credential.id.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


def _mask(email: str) -> str:
    name, _, domain = email.partition("@")
    head = name[:2] if len(name) > 2 else name[:1]
    return f"{head}***@{domain}" if domain else f"{head}***"
