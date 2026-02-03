import hashlib
import sqlite3
import time
from pathlib import Path


DB_PATH = Path("data/auth.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Default users seeded on first run (plain passwords — stored hashed)
_SEED_USERS = [
    ("admin", "admin123"),
    ("commandcentre", "command@2024"),
    ("command_admin", "iris_admin#2024"),
]


def _hash_password(password: str) -> str:
    """SHA-256 hash for password storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _get_conn():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def init_db():
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT    NOT NULL UNIQUE,
        password TEXT    NOT NULL,
        created  INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS login_events (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT    NOT NULL,
        success   INTEGER NOT NULL,
        timestamp INTEGER NOT NULL
    )
    """)

    # Seed default users if table is empty
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        now = int(time.time())
        for uname, pwd in _SEED_USERS:
            cur.execute(
                "INSERT INTO users (username, password, created) VALUES (?, ?, ?)",
                (uname, _hash_password(pwd), now),
            )
        print(f"[AUTH] Seeded {len(_SEED_USERS)} default users")

    conn.commit()
    conn.close()


def log_login(username: str, success: bool):
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO login_events (username, success, timestamp) VALUES (?, ?, ?)",
            (username, int(success), int(time.time())),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def login_user(username: str, password: str) -> dict:
    """Validate credentials against the users table."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    success = row is not None and row[0] == _hash_password(password)
    log_login(username, success)

    if success:
        return {"success": True, "username": username}
    return {"success": False, "error": "Invalid credentials"}


def add_user(username: str, password: str) -> dict:
    """Add a new user. Returns success/error."""
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password, created) VALUES (?, ?, ?)",
            (username, _hash_password(password), int(time.time())),
        )
        conn.commit()
        return {"success": True, "username": username}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Username already exists"}
    finally:
        conn.close()


def delete_user(username: str) -> dict:
    """Delete a user by username."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE username = ?", (username,))
    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    if deleted:
        return {"success": True}
    return {"success": False, "error": "User not found"}


def change_password(username: str, new_password: str) -> dict:
    """Update a user's password."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET password = ? WHERE username = ?",
        (_hash_password(new_password), username),
    )
    updated = cur.rowcount > 0
    conn.commit()
    conn.close()
    if updated:
        return {"success": True}
    return {"success": False, "error": "User not found"}


def list_users() -> list:
    """Return all usernames and creation timestamps."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, created FROM users ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [{"username": r[0], "created": r[1]} for r in rows]


# Auto-init on import
init_db()
