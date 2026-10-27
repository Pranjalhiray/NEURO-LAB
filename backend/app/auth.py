"""Short-lived in-memory admin sessions for the local research dashboard."""
import hashlib
import hmac
import os
import secrets
import time

from fastapi import Header, HTTPException

ADMIN_SESSION_SECONDS = 8 * 60 * 60
_sessions: dict[str, float] = {}


def create_admin_session(username: str, password: str) -> str:
    expected_username = os.getenv("BTDS_ADMIN_USERNAME", "")
    expected_password = os.getenv("BTDS_ADMIN_PASSWORD", "")
    if not expected_username or not expected_password or expected_password == "replace-with-a-long-random-password":
        raise HTTPException(status_code=503, detail="Admin credentials are not configured. Set BTDS_ADMIN_USERNAME and BTDS_ADMIN_PASSWORD in the local .env file.")
    if not hmac.compare_digest(username.encode("utf-8"), expected_username.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Username or password is incorrect.")
    if not hmac.compare_digest(password.encode("utf-8"), expected_password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Username or password is incorrect.")
    now = time.time()
    for key, expiry in list(_sessions.items()):
        if expiry <= now:
            _sessions.pop(key, None)
    token = secrets.token_urlsafe(32)
    _sessions[hashlib.sha256(token.encode("ascii")).hexdigest()] = now + ADMIN_SESSION_SECONDS
    return token


def require_admin(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Administrator sign-in required.")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Administrator sign-in required.")
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expiry = _sessions.get(digest)
    if expiry is None or expiry <= time.time():
        _sessions.pop(digest, None)
        raise HTTPException(status_code=401, detail="Administrator session expired. Please sign in again.")
    return os.getenv("BTDS_ADMIN_USERNAME", "admin")


def active_admin_sessions_count() -> int:
    now = time.time()
    for key, expiry in list(_sessions.items()):
        if expiry <= now:
            _sessions.pop(key, None)
    return len(_sessions)


def revoke_admin_session(token: str) -> None:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    _sessions.pop(digest, None)
