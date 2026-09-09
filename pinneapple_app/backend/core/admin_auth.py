"""pinneapple_app.backend.core.admin_auth — a minimal, generic admin gate.

Deliberately a single shared bearer token from an environment variable
(``PINNEAPPLE_ADMIN_TOKEN``), NOT an email-domain allowlist: this is an
open-source library's reference app, not a single company's internal
deployment, so gating admin access to "@some-company.com" would not
generalize to anyone else running this app. Anyone deploying this for
real, multi-user access should put a real auth provider in front of it —
this is intentionally the simplest thing that is still actually a gate
(not "no gate at all"), matching the honesty this repo's other modules
apply to their own scope limits.
"""
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException


def require_admin(x_admin_token: str = Header(default="")) -> None:
    """FastAPI dependency: raises 401 unless ``X-Admin-Token`` matches
    ``PINNEAPPLE_ADMIN_TOKEN``. If that env var is unset, admin routes are
    refused entirely (503) rather than silently open — there is no
    "admin token not configured, so let everyone in" fallback."""
    expected = os.environ.get("PINNEAPPLE_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="PINNEAPPLE_ADMIN_TOKEN is not configured on this server.")
    if not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token header.")
