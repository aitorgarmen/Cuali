"""
Authentication helpers for the PyQt application.

Implements secure password hashing (PBKDF2-HMAC-SHA256) without external
dependencies and minimal DB helpers using the existing BatchDB instance.

Table created: app_user

Columns:
- id SERIAL PRIMARY KEY
- username TEXT UNIQUE NOT NULL
- password TEXT NOT NULL   # format: pbkdf2_sha256$iterations$salt_b64$hash_b64
- role TEXT NOT NULL DEFAULT 'user'   # e.g. 'admin', 'operador', 'viewer'
- is_active BOOLEAN NOT NULL DEFAULT TRUE
- created_at TIMESTAMP NOT NULL DEFAULT NOW()
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Optional, Dict, Any


PBKDF2_ALGO = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 200_000
PBKDF2_SALT_BYTES = 16


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    if not isinstance(password, str) or password == "":
        raise ValueError("La contraseña no puede estar vacía")
    salt = secrets.token_bytes(PBKDF2_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{PBKDF2_ALGO}${iterations}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iter_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != PBKDF2_ALGO:
            return False
        iters = int(iter_s)
        salt = _b64d(salt_b64)
        expected = _b64d(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# Simple in-process mock store for when DB connection is mocked
_MOCK_USERS: Dict[str, Dict[str, Any]] = {}


def ensure_auth_tables(db) -> None:
    """Create the app_user table if it does not exist."""
    if getattr(db, "_mock", False):
        return
    sql = (
        "CREATE TABLE IF NOT EXISTS app_user ("
        "  id SERIAL PRIMARY KEY,"
        "  username TEXT UNIQUE NOT NULL,"
        "  password TEXT NOT NULL,"
        "  role TEXT NOT NULL DEFAULT 'user',"
        "  is_active BOOLEAN NOT NULL DEFAULT TRUE,"
        "  created_at TIMESTAMP NOT NULL DEFAULT NOW()"
        ")"
    )
    db.cur.execute(sql)
    # Ensure unique index on lower(username) to avoid case duplicates
    db.cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_app_user_username_lower "
        "ON app_user ((lower(username)))"
    )


def count_users(db) -> int:
    if getattr(db, "_mock", False):
        return sum(1 for u in _MOCK_USERS.values() if u.get("is_active", True))
    db.cur.execute("SELECT COUNT(*) FROM app_user WHERE is_active")
    row = db.cur.fetchone()
    return int(row[0]) if row else 0


def get_user(db, username: str) -> Optional[Dict[str, Any]]:
    uname = username.strip()
    if uname == "":
        return None
    if getattr(db, "_mock", False):
        u = _MOCK_USERS.get(uname.lower())
        return dict(u) if u else None
    db.cur.execute(
        "SELECT id, username, password, role, is_active FROM app_user WHERE lower(username)=lower(%s)",
        (uname,)
    )
    row = db.cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "username": row[1],
        "password": row[2],
        "role": row[3],
        "is_active": row[4],
    }


ALLOWED_ROLES = ("admin", "cualificacion", "baseline")


def create_user(db, username: str, password: str, role: str = "baseline") -> Dict[str, Any]:
    uname = username.strip()
    if uname == "":
        raise ValueError("El nombre de usuario no puede estar vacío")
    if len(password) < 6:
        raise ValueError("La contraseña debe tener al menos 6 caracteres")
    if role not in ALLOWED_ROLES:
        role = "baseline"
    pwd = hash_password(password)
    if getattr(db, "_mock", False):
        key = uname.lower()
        if key in _MOCK_USERS:
            raise ValueError("El usuario ya existe")
        u = {"id": len(_MOCK_USERS) + 1, "username": uname, "password": pwd, "role": role, "is_active": True}
        _MOCK_USERS[key] = u
        return dict(u)
    # Try insert
    db.cur.execute(
        "INSERT INTO app_user (username, password, role) VALUES (%s, %s, %s) "
        "ON CONFLICT ((lower(username))) DO NOTHING RETURNING id",
        (uname, pwd, role),
    )
    row = db.cur.fetchone()
    if not row:
        # if conflict, fetch and error
        raise ValueError("El usuario ya existe")
    return {"id": int(row[0]), "username": uname, "password": pwd, "role": role, "is_active": True}


def authenticate(db, username: str, password: str) -> Optional[Dict[str, Any]]:
    u = get_user(db, username)
    if not u or not u.get("is_active", True):
        return None
    stored = u.get("password") or ""
    if verify_password(password, stored):
        # don't leak hash in returned dict
        return {k: v for k, v in u.items() if k != "password"}
    return None
