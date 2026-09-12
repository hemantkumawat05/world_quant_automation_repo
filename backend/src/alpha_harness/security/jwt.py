"""JWT token generation and verification.

Zero extra dependencies: uses standard hmac, hashlib, base64, and json.
Signs with HS256 using the master key from Vault.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


class JWTError(Exception):
    """Raised when token verification fails."""


def create_access_token(
    user_id: str,
    email: str,
    key: bytes,
    expires_in_seconds: int = 60 * 60 * 24 * 30,  # 30 days default
) -> str:
    """Create a signed HS256 JWT access token."""
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + expires_in_seconds,
    }

    header_b64 = _b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    signature = hmac.new(key, signing_input, hashlib.sha256).digest()
    sig_b64 = _b64encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def verify_access_token(token: str, key: bytes) -> dict[str, Any]:
    """Verify an HS256 JWT access token and return the payload."""
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise JWTError("malformed_token")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    try:
        expected_sig = hmac.new(key, signing_input, hashlib.sha256).digest()
        actual_sig = _b64decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise JWTError("invalid_signature")
    except Exception as exc:
        raise JWTError("invalid_signature") from exc

    try:
        payload_bytes = _b64decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise JWTError("invalid_payload") from exc

    exp = payload.get("exp")
    if exp is not None and time.time() > float(exp):
        raise JWTError("token_expired")

    if not payload.get("sub"):
        raise JWTError("missing_subject")

    return payload
