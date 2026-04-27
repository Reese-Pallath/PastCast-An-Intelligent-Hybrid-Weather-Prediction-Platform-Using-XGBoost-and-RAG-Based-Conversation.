from __future__ import annotations

"""
Enhanced Database Layer — Session-aware chat with LSTM state persistence.

Production improvements over original:
- WAL mode for concurrent read/write
- Database indices for query performance
- Session cleanup/expiry utility
- Structured logging instead of silent failures
"""

import logging
import os
import sqlite3
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DB = os.getenv("DB_PATH", "chat_memory.db")


def _connect():
    conn = sqlite3.connect(DB)
    # Enable WAL mode for better concurrent read/write performance
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    conn = _connect()
    c = conn.cursor()

    # Original chat history table (preserved for backward compat)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Session management table
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
            lstm_state BLOB,
            context_summary TEXT DEFAULT ''
        )
    """)

    # Add session_id column to chat_history if missing (migration)
    try:
        c.execute("SELECT session_id FROM chat_history LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT")

    # ── Indices for query performance ───────────────────────────
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_session_id
        ON chat_history(session_id)
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp
        ON chat_history(timestamp)
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_session_id_id
        ON chat_history(session_id, id)
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_active
        ON chat_sessions(last_active)
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised (WAL mode, indices created)")


# ── Session Management ──────────────────────────────────────────

def create_session() -> str:
    """Create a new chat session. Returns session_id."""
    session_id = str(uuid.uuid4())
    conn = _connect()
    conn.execute(
        "INSERT INTO chat_sessions (session_id) VALUES (?)",
        (session_id,)
    )
    conn.commit()
    conn.close()
    logger.debug("Session created: %s", session_id)
    return session_id


def get_session(session_id: str) -> dict | None:
    """Get session details."""
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT session_id, created_at, last_active, context_summary "
        "FROM chat_sessions WHERE session_id = ?",
        (session_id,)
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "session_id": row[0],
        "created_at": row[1],
        "last_active": row[2],
        "context_summary": row[3] or "",
    }


def touch_session(session_id: str):
    """Update last_active timestamp."""
    conn = _connect()
    conn.execute(
        "UPDATE chat_sessions SET last_active = ? WHERE session_id = ?",
        (datetime.now().isoformat(), session_id)
    )
    conn.commit()
    conn.close()


def save_lstm_state(session_id: str, state_bytes: bytes):
    """Save serialised LSTM hidden state for a session."""
    conn = _connect()
    conn.execute(
        "UPDATE chat_sessions SET lstm_state = ?, last_active = ? WHERE session_id = ?",
        (state_bytes, datetime.now().isoformat(), session_id)
    )
    conn.commit()
    conn.close()


def load_lstm_state(session_id: str) -> bytes | None:
    """Load serialised LSTM hidden state for a session."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT lstm_state FROM chat_sessions WHERE session_id = ?",
              (session_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def update_context_summary(session_id: str, summary: str):
    """Update the context summary for a session."""
    conn = _connect()
    conn.execute(
        "UPDATE chat_sessions SET context_summary = ? WHERE session_id = ?",
        (summary, session_id)
    )
    conn.commit()
    conn.close()


# ── Message Management ──────────────────────────────────────────

def add_message(role: str, content: str, session_id: str = None):
    """Add a message to chat history."""
    conn = _connect()
    conn.execute(
        "INSERT INTO chat_history (role, content, session_id) VALUES (?, ?, ?)",
        (role, content, session_id)
    )
    conn.commit()
    conn.close()


def get_recent_messages(limit: int = 6, session_id: str = None):
    """Get recent messages, optionally filtered by session."""
    conn = _connect()
    c = conn.cursor()
    if session_id:
        c.execute(
            "SELECT role, content FROM chat_history "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        )
    else:
        c.execute(
            "SELECT role, content FROM chat_history ORDER BY id DESC LIMIT ?",
            (limit,)
        )
    rows = c.fetchall()
    conn.close()
    return rows[::-1]


def get_session_messages(session_id: str, limit: int = 50) -> list:
    """Get all messages for a session (for LSTM processing)."""
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT role, content, timestamp FROM chat_history "
        "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c, "timestamp": t} for r, c, t in rows]


def clear_history(session_id: str = None):
    """Clear chat history. If session_id given, clear only that session."""
    conn = _connect()
    if session_id:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    else:
        conn.execute("DELETE FROM chat_history")
        conn.execute("DELETE FROM chat_sessions")
    conn.commit()
    conn.close()


# ── Session Cleanup ─────────────────────────────────────────────

def cleanup_old_sessions(max_age_days: int = 30) -> int:
    """
    Delete sessions and their messages older than max_age_days.

    Returns the number of sessions deleted.
    """
    cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
    conn = _connect()
    c = conn.cursor()

    # Find expired sessions
    c.execute(
        "SELECT session_id FROM chat_sessions WHERE last_active < ?",
        (cutoff,)
    )
    expired = [row[0] for row in c.fetchall()]

    if expired:
        placeholders = ",".join("?" * len(expired))
        conn.execute(
            f"DELETE FROM chat_history WHERE session_id IN ({placeholders})",
            expired,
        )
        conn.execute(
            f"DELETE FROM chat_sessions WHERE session_id IN ({placeholders})",
            expired,
        )
        conn.commit()
        logger.info("Cleaned up %d expired sessions (older than %d days)", len(expired), max_age_days)

    conn.close()
    return len(expired)
