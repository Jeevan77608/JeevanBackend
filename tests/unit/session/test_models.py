"""Tests for src/session/models.py — SessionState, SessionFingerprint,
ConversationMessage, ConversationHistory."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from src.session.models import (
    ConversationHistory,
    ConversationMessage,
    SessionFingerprint,
    SessionState,
)


# =============================================================================
# SessionState
# =============================================================================


class TestSessionState:
    """Tests for the SessionState enum."""

    def test_active_value(self) -> None:
        assert SessionState.ACTIVE == "active"

    def test_expired_value(self) -> None:
        assert SessionState.EXPIRED == "expired"

    def test_revoked_value(self) -> None:
        assert SessionState.REVOKED == "revoked"

    def test_locked_value(self) -> None:
        assert SessionState.LOCKED == "locked"

    def test_all_members_present(self) -> None:
        names = {s.name for s in SessionState}
        assert names == {"ACTIVE", "EXPIRED", "REVOKED", "LOCKED"}


# =============================================================================
# SessionFingerprint
# =============================================================================


class TestSessionFingerprint:
    """Tests for the SessionFingerprint model."""

    def test_create_produces_hash(self) -> None:
        fp = SessionFingerprint.create(user_agent="Mozilla/5.0")
        expected = hashlib.sha256("Mozilla/5.0".encode()).hexdigest()
        assert fp.user_agent_hash == expected

    def test_create_without_ip_leaves_ip_hash_none(self) -> None:
        fp = SessionFingerprint.create(user_agent="UA")
        assert fp.ip_hash is None

    def test_create_with_ip_hashes_ip(self) -> None:
        fp = SessionFingerprint.create(user_agent="UA", client_ip="10.0.0.1")
        expected_ip = hashlib.sha256("10.0.0.1".encode()).hexdigest()
        assert fp.ip_hash == expected_ip

    def test_create_with_accept_language(self) -> None:
        fp = SessionFingerprint.create(
            user_agent="UA",
            accept_language="en-US,en;q=0.9",
        )
        assert fp.accept_language == "en-US,en;q=0.9"

    def test_same_user_agent_gives_same_hash(self) -> None:
        fp1 = SessionFingerprint.create(user_agent="Same-UA")
        fp2 = SessionFingerprint.create(user_agent="Same-UA")
        assert fp1.user_agent_hash == fp2.user_agent_hash

    def test_different_user_agent_gives_different_hash(self) -> None:
        fp1 = SessionFingerprint.create(user_agent="UA-1")
        fp2 = SessionFingerprint.create(user_agent="UA-2")
        assert fp1.user_agent_hash != fp2.user_agent_hash

    def test_matches_same_user_agent(self) -> None:
        fp1 = SessionFingerprint.create(user_agent="Mozilla/5.0")
        fp2 = SessionFingerprint.create(user_agent="Mozilla/5.0")
        assert fp1.matches(fp2) is True

    def test_matches_different_user_agent_fails(self) -> None:
        fp1 = SessionFingerprint.create(user_agent="Mozilla/5.0")
        fp2 = SessionFingerprint.create(user_agent="Chrome/120")
        assert fp1.matches(fp2) is False

    def test_matches_strict_ip_same_ip(self) -> None:
        fp1 = SessionFingerprint.create(user_agent="UA", client_ip="10.0.0.1")
        fp2 = SessionFingerprint.create(user_agent="UA", client_ip="10.0.0.1")
        assert fp1.matches(fp2, strict_ip=True) is True

    def test_matches_strict_ip_different_ip_fails(self) -> None:
        fp1 = SessionFingerprint.create(user_agent="UA", client_ip="10.0.0.1")
        fp2 = SessionFingerprint.create(user_agent="UA", client_ip="192.168.1.1")
        assert fp1.matches(fp2, strict_ip=True) is False

    def test_matches_non_strict_ip_different_ip_passes(self) -> None:
        fp1 = SessionFingerprint.create(user_agent="UA", client_ip="10.0.0.1")
        fp2 = SessionFingerprint.create(user_agent="UA", client_ip="192.168.1.1")
        assert fp1.matches(fp2, strict_ip=False) is True


# =============================================================================
# ConversationMessage
# =============================================================================


class TestConversationMessage:
    """Tests for the ConversationMessage model."""

    def test_valid_role_user(self) -> None:
        msg = ConversationMessage(role="user", content="Hello")
        assert msg.role == "user"

    def test_valid_role_assistant(self) -> None:
        msg = ConversationMessage(role="assistant", content="Hi")
        assert msg.role == "assistant"

    def test_valid_role_system(self) -> None:
        msg = ConversationMessage(role="system", content="You are a helper.")
        assert msg.role == "system"

    def test_valid_role_tool(self) -> None:
        msg = ConversationMessage(role="tool", content="result")
        assert msg.role == "tool"

    def test_invalid_role_raises_value_error(self) -> None:
        with pytest.raises(ValidationError, match="Role must be one of"):
            ConversationMessage(role="admin", content="x")

    def test_default_metadata_empty(self) -> None:
        msg = ConversationMessage(role="user", content="hi")
        assert msg.metadata == {}

    def test_timestamp_auto_set(self) -> None:
        msg = ConversationMessage(role="user", content="hi")
        assert msg.timestamp is not None


# =============================================================================
# ConversationHistory
# =============================================================================


class TestConversationHistory:
    """Tests for the ConversationHistory model."""

    def test_add_message_appends(self) -> None:
        history = ConversationHistory(session_id="s1")
        history.add_message("user", "Hello")
        assert len(history.messages) == 1
        assert history.messages[0].content == "Hello"

    def test_add_message_updates_updated_at(self) -> None:
        history = ConversationHistory(session_id="s1")
        old_ts = history.updated_at
        history.add_message("user", "Hello")
        assert history.updated_at >= old_ts

    def test_trimming_keeps_system_messages(self) -> None:
        history = ConversationHistory(session_id="s1", max_messages=5)
        history.add_message("system", "System instruction")
        for i in range(6):
            history.add_message("user", f"msg-{i}")
        # After trimming we should have at most max_messages
        assert len(history.messages) <= 5
        # System message must survive
        system_msgs = [m for m in history.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "System instruction"

    def test_trimming_removes_oldest_non_system(self) -> None:
        history = ConversationHistory(session_id="s1", max_messages=3)
        history.add_message("user", "first")
        history.add_message("user", "second")
        history.add_message("user", "third")
        history.add_message("user", "fourth")
        # "first" should have been trimmed
        contents = [m.content for m in history.messages]
        assert "first" not in contents
        assert "fourth" in contents

    def test_get_context_window_respects_char_limit(self) -> None:
        history = ConversationHistory(session_id="s1")
        # Each message is 100 chars
        for i in range(10):
            history.add_message("user", "x" * 100)
        # max_tokens=50 => max_chars=200 => at most 2 messages fit
        window = history.get_context_window(max_tokens=50)
        assert len(window) <= 2

    def test_get_context_window_returns_most_recent(self) -> None:
        history = ConversationHistory(session_id="s1")
        history.add_message("user", "old")
        history.add_message("user", "new")
        # Enough room for both
        window = history.get_context_window(max_tokens=4000)
        assert window[-1]["content"] == "new"

    def test_get_context_window_format(self) -> None:
        history = ConversationHistory(session_id="s1")
        history.add_message("user", "Hello")
        window = history.get_context_window()
        assert window == [{"role": "user", "content": "Hello"}]

    def test_clear_keeps_only_system_messages(self) -> None:
        history = ConversationHistory(session_id="s1")
        history.add_message("system", "persist me")
        history.add_message("user", "remove me")
        history.add_message("assistant", "also remove")
        history.clear()
        assert len(history.messages) == 1
        assert history.messages[0].role == "system"

    def test_clear_on_empty_history(self) -> None:
        history = ConversationHistory(session_id="s1")
        history.clear()
        assert history.messages == []
