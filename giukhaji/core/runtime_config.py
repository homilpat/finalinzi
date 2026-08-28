"""Production-safe runtime secrets with local development fallbacks."""

from __future__ import annotations

import os
import secrets


def _secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if os.environ.get("RENDER", "").lower() == "true":
        raise RuntimeError(f"{name} must be configured in production")
    return secrets.token_urlsafe(32)


SECRET_KEY = _secret("SECRET_KEY")
PHONE_HASH_SALT = _secret("PHONE_HASH_SALT")

