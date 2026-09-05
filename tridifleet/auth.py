from __future__ import annotations

import os
import secrets
from fastapi import HTTPException, Request, Response, WebSocket, status


COOKIE_NAME = "tridifleet_session"
ADMIN_USERNAME = os.getenv("TRIDIFLEET_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("TRIDIFLEET_ADMIN_PASSWORD", "tridifleet-local")
_sessions: set[str] = set()


def login(username: str, password: str, response: Response) -> bool:
    ok = secrets.compare_digest(username, ADMIN_USERNAME) and secrets.compare_digest(
        password, ADMIN_PASSWORD
    )
    if not ok:
        return False
    token = secrets.token_urlsafe(32)
    _sessions.add(token)
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="strict",
        secure=False,
        max_age=60 * 60 * 24 * 30,
    )
    return True


def logout(request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        _sessions.discard(token)
    response.delete_cookie(COOKIE_NAME)


def require_admin(request: Request) -> str:
    token = request.cookies.get(COOKIE_NAME)
    if not token or token not in _sessions:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="login required")
    return ADMIN_USERNAME


def websocket_is_admin(websocket: WebSocket) -> bool:
    token = websocket.cookies.get(COOKIE_NAME)
    return bool(token and token in _sessions)
