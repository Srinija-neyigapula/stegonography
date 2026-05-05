"""Authentication helpers – Emergent-managed Google OAuth."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import HTTPException, Request

EMERGENT_AUTH_URL = "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data"
SESSION_DURATION_DAYS = 7


def _new_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:12]}"


async def fetch_emergent_session(session_id: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            EMERGENT_AUTH_URL, headers={"X-Session-ID": session_id}
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid session_id")
    return resp.json()


async def upsert_user(db, profile: dict) -> dict:
    existing = await db.users.find_one({"email": profile["email"]}, {"_id": 0})
    if existing:
        await db.users.update_one(
            {"email": profile["email"]},
            {"$set": {
                "name": profile.get("name"),
                "picture": profile.get("picture"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }},
        )
        return existing
    user = {
        "user_id": _new_user_id(),
        "email": profile["email"],
        "name": profile.get("name"),
        "picture": profile.get("picture"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.users.insert_one(user.copy())
    return user


async def store_session(db, user_id: str, session_token: str) -> datetime:
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_DURATION_DAYS)
    await db.user_sessions.insert_one({
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return expires_at


async def get_user_from_request(db, request: Request) -> Optional[dict]:
    token = request.cookies.get("session_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        return None

    session = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if not session:
        return None

    expires_at = session["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        return None

    user = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0})
    return user


async def delete_session(db, session_token: str) -> None:
    await db.user_sessions.delete_one({"session_token": session_token})
