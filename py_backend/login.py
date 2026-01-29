from pathlib import Path
import sqlite3
import time

from pydantic import BaseModel


# ============================
# HARD-CODED CREDENTIALS
# ============================

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


# ============================
# OPTIONAL SQLITE (FOR LOGGING)
# ============================

DB_PATH = Path("data/auth.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS login_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        success INTEGER NOT NULL,
        timestamp INTEGER NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def log_login(username: str, success: bool):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO login_events (username, success, timestamp) VALUES (?, ?, ?)",
            (username, int(success), int(time.time()))
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # logging must never block login


# ============================
# REQUEST MODEL
# ============================

class LoginRequest(BaseModel):
    username: str
    password: str


# ============================
# LOGIN LOGIC (UI CONTRACT)
# ============================

def login_user(username: str, password: str) -> dict:
    success = (username == ADMIN_USERNAME and password == ADMIN_PASSWORD)

    log_login(username, success)

    if success:
        return {
            "success": True
        }

    return {
        "success": False,
        "error": "Invalid credentials"
    }


# ============================
# INIT
# ============================

init_db()
