"""Tests for src/session/redis_keys.py — RedisKeys classmethods."""

from __future__ import annotations

from src.session.redis_keys import RedisKeys


class TestRedisKeys:
    """Tests for all RedisKeys classmethods."""

    def test_session_key(self) -> None:
        assert RedisKeys.session("abc-123") == "session:abc-123"

    def test_session_key_with_special_chars(self) -> None:
        assert RedisKeys.session("s-1/2") == "session:s-1/2"

    def test_drafts_key(self) -> None:
        assert RedisKeys.drafts("sess-42") == "drafts:sess-42"

    def test_conversation_key(self) -> None:
        assert RedisKeys.conversation("sess-99") == "conversation:sess-99"

    def test_draft_conversation_key(self) -> None:
        assert RedisKeys.draft_conversation("sess-dc") == "draft_conversation:sess-dc"

    def test_auth_state_key(self) -> None:
        assert RedisKeys.auth_state("state-xyz") == "auth_state:state-xyz"

    def test_user_sessions_key(self) -> None:
        assert RedisKeys.user_sessions("user-007") == "user_sessions:user-007"

    def test_file_content_key(self) -> None:
        assert RedisKeys.file_content("sess-1", "file-a") == "file:sess-1:file-a"

    def test_board_context_key(self) -> None:
        assert RedisKeys.board_context("sess-1", "proj-x") == "board_context:sess-1:proj-x"

    def test_keys_use_correct_separators(self) -> None:
        """Verify colon-separated patterns match the class-level constants."""
        assert RedisKeys.SESSION == "session:{session_id}"
        assert RedisKeys.DRAFTS == "drafts:{session_id}"
        assert RedisKeys.CONVERSATION == "conversation:{session_id}"
        assert RedisKeys.DRAFT_CONVERSATION == "draft_conversation:{session_id}"
        assert RedisKeys.AUTH_STATE == "auth_state:{state}"
        assert RedisKeys.USER_SESSIONS == "user_sessions:{user_id}"
        assert RedisKeys.FILE_CONTENT == "file:{session_id}:{file_id}"
        assert RedisKeys.BOARD_CONTEXT == "board_context:{session_id}:{project_id}"

    def test_session_key_empty_string(self) -> None:
        assert RedisKeys.session("") == "session:"

    def test_file_content_key_with_uuid_ids(self) -> None:
        key = RedisKeys.file_content(
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        )
        assert key == "file:550e8400-e29b-41d4-a716-446655440000:6ba7b810-9dad-11d1-80b4-00c04fd430c8"
