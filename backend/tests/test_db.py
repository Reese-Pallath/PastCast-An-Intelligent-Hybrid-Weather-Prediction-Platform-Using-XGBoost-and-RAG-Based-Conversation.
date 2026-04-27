"""
Database layer tests for PastCast.
Tests session CRUD, message storage, and cleanup.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.db import (
    init_db, add_message, get_recent_messages, clear_history,
    create_session, get_session, touch_session,
    cleanup_old_sessions, get_session_messages,
)


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("utils.db.DB", db_path)
    init_db()
    yield


class TestSessionManagement:

    def test_create_session_returns_uuid(self):
        sid = create_session()
        assert len(sid) == 36  # UUID format
        assert "-" in sid

    def test_get_session_returns_details(self):
        sid = create_session()
        session = get_session(sid)
        assert session is not None
        assert session["session_id"] == sid
        assert "created_at" in session
        assert "last_active" in session

    def test_get_nonexistent_session_returns_none(self):
        session = get_session("nonexistent-id")
        assert session is None

    def test_touch_session_updates_last_active(self):
        sid = create_session()
        s1 = get_session(sid)
        touch_session(sid)
        s2 = get_session(sid)
        # last_active should be updated (or at least not earlier)
        assert s2["last_active"] >= s1["last_active"]


class TestMessageManagement:

    def test_add_and_retrieve_message(self):
        sid = create_session()
        add_message("user", "Hello!", session_id=sid)
        add_message("ai", "Hi there!", session_id=sid)
        msgs = get_recent_messages(10, session_id=sid)
        assert len(msgs) == 2
        assert msgs[0] == ("user", "Hello!")
        assert msgs[1] == ("ai", "Hi there!")

    def test_get_recent_messages_respects_limit(self):
        sid = create_session()
        for i in range(10):
            add_message("user", f"Message {i}", session_id=sid)
        msgs = get_recent_messages(5, session_id=sid)
        assert len(msgs) == 5

    def test_messages_filtered_by_session(self):
        sid1 = create_session()
        sid2 = create_session()
        add_message("user", "Session 1 message", session_id=sid1)
        add_message("user", "Session 2 message", session_id=sid2)
        msgs1 = get_recent_messages(10, session_id=sid1)
        msgs2 = get_recent_messages(10, session_id=sid2)
        assert len(msgs1) == 1
        assert len(msgs2) == 1
        assert msgs1[0][1] == "Session 1 message"
        assert msgs2[0][1] == "Session 2 message"

    def test_get_session_messages_returns_dicts(self):
        sid = create_session()
        add_message("user", "Test", session_id=sid)
        msgs = get_session_messages(sid)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Test"
        assert "timestamp" in msgs[0]


class TestClearHistory:

    def test_clear_specific_session(self):
        sid1 = create_session()
        sid2 = create_session()
        add_message("user", "Keep this", session_id=sid2)
        add_message("user", "Delete this", session_id=sid1)
        clear_history(session_id=sid1)
        assert get_session(sid1) is None
        assert get_session(sid2) is not None
        msgs = get_recent_messages(10, session_id=sid2)
        assert len(msgs) == 1

    def test_clear_all(self):
        sid1 = create_session()
        sid2 = create_session()
        add_message("user", "Msg 1", session_id=sid1)
        add_message("user", "Msg 2", session_id=sid2)
        clear_history()
        assert get_session(sid1) is None
        assert get_session(sid2) is None


class TestCleanup:

    def test_cleanup_removes_nothing_when_fresh(self):
        create_session()
        count = cleanup_old_sessions(max_age_days=30)
        assert count == 0

    def test_cleanup_removes_old_sessions(self, monkeypatch):
        """Simulate old sessions by setting last_active in the past."""
        import sqlite3
        from datetime import datetime, timedelta

        sid = create_session()
        add_message("user", "Old message", session_id=sid)

        # Manually set last_active to 60 days ago
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        import utils.db as db_mod
        conn = sqlite3.connect(db_mod.DB)
        conn.execute(
            "UPDATE chat_sessions SET last_active = ? WHERE session_id = ?",
            (old_date, sid)
        )
        conn.commit()
        conn.close()

        count = cleanup_old_sessions(max_age_days=30)
        assert count == 1
        assert get_session(sid) is None
