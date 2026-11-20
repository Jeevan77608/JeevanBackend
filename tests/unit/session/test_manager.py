"""Tests for src/session/manager.py -- SessionManager."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.config.settings import Settings, clear_settings_cache
from src.models.user import SessionContext, TokenInfo, UserProfile
from src.session.exceptions import (
    SessionExpiredError,
    SessionFingerprintMismatchError,
    SessionNotFoundError,
)
from src.session.manager import SessionManager
from src.session.models import SessionFingerprint
from src.session.redis_keys import RedisKeys


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
async def manager(mock_settings: Settings):
    """Initialize a SessionManager with in-memory store."""
    mgr = SessionManager(mock_settings, enable_fingerprinting=True)
    await mgr.initialize()
    yield mgr
    await mgr.close()


@pytest.fixture()
def fingerprint_a() -> SessionFingerprint:
    """A deterministic fingerprint for testing."""
    return SessionFingerprint.create(
        user_agent="Mozilla/5.0 TestAgent",
        client_ip="10.0.0.1",
        accept_language="en-US",
    )


@pytest.fixture()
def fingerprint_b() -> SessionFingerprint:
    """A different fingerprint (different user-agent) to trigger mismatch."""
    return SessionFingerprint.create(
        user_agent="curl/7.68.0 DifferentAgent",
        client_ip="10.0.0.2",
        accept_language="de-DE",
    )


# =============================================================================
# Lifecycle
# =============================================================================


class TestLifecycle:
    """Tests for manager initialization and teardown."""

    async def test_initialize_creates_repositories(
        self, manager: SessionManager
    ) -> None:
        """After initialization all repository properties are available."""
        assert manager.drafts is not None
        assert manager.conversations is not None
        assert manager.files is not None
        assert manager.board_context_store is not None
        assert manager.auth_state_store is not None

    async def test_close_succeeds(self, mock_settings: Settings) -> None:
        """close() completes without error."""
        mgr = SessionManager(mock_settings)
        await mgr.initialize()
        await mgr.close()  # should not raise

    async def test_properties_fail_before_initialize(
        self, mock_settings: Settings
    ) -> None:
        """Accessing a repository before initialize() raises AssertionError."""
        mgr = SessionManager(mock_settings)
        with pytest.raises(AssertionError, match="not initialized"):
            _ = mgr.drafts


# =============================================================================
# Session ID / State Generation
# =============================================================================


class TestSessionIdGeneration:
    """Tests for static ID generators."""

    def test_generate_session_id_unique(self) -> None:
        """Two consecutive calls produce distinct IDs."""
        id_a = SessionManager.generate_session_id()
        id_b = SessionManager.generate_session_id()
        assert id_a != id_b

    def test_generate_session_id_length(self) -> None:
        """Generated session ID is at least 32 characters long."""
        sid = SessionManager.generate_session_id()
        assert len(sid) >= 32

    def test_generate_state_unique(self) -> None:
        """Two consecutive state values are distinct."""
        s_a = SessionManager.generate_state()
        s_b = SessionManager.generate_state()
        assert s_a != s_b


# =============================================================================
# Session CRUD
# =============================================================================


class TestCreateSession:
    """Tests for session creation."""

    async def test_create_session_returns_session_context(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """create_session returns a SessionContext with expected fields."""
        session = await manager.create_session(sample_user, sample_token_info)
        assert isinstance(session, SessionContext)
        assert session.session_id
        assert session.user.id == sample_user.id
        assert session.token_info.access_token == sample_token_info.access_token

    async def test_create_session_generates_unique_ids(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """Two create_session calls produce different session_ids."""
        s1 = await manager.create_session(sample_user, sample_token_info)
        s2 = await manager.create_session(sample_user, sample_token_info)
        assert s1.session_id != s2.session_id


class TestGetSession:
    """Tests for session retrieval."""

    async def test_get_session_success(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """A created session can be retrieved by its ID."""
        created = await manager.create_session(sample_user, sample_token_info)
        fetched = await manager.get_session(created.session_id)
        assert fetched.session_id == created.session_id
        assert fetched.user.id == sample_user.id

    async def test_get_session_not_found_raises(
        self, manager: SessionManager
    ) -> None:
        """Fetching a non-existent session raises SessionNotFoundError."""
        with pytest.raises(SessionNotFoundError):
            await manager.get_session("totally-bogus-session-id")

    async def test_get_session_extends_expiry_by_default(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """get_session with default extend_expiry=True updates last_accessed."""
        created = await manager.create_session(sample_user, sample_token_info)
        original_last_accessed = created.last_accessed

        fetched = await manager.get_session(created.session_id)
        # last_accessed should be >= original (re-saved on extend)
        assert fetched.last_accessed >= original_last_accessed

    async def test_get_session_no_extend_when_disabled(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """get_session with extend_expiry=False does not re-save session."""
        created = await manager.create_session(sample_user, sample_token_info)

        # Read the raw stored data before get_session
        store = manager._get_store()
        key = RedisKeys.session(created.session_id)
        raw_before = await store.get(key)

        fetched = await manager.get_session(
            created.session_id, extend_expiry=False
        )

        raw_after = await store.get(key)
        # The raw stored data should be identical because no re-save occurred
        assert raw_before == raw_after

    async def test_get_session_decrypts_token(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """Decrypted token_info.access_token matches the original."""
        created = await manager.create_session(sample_user, sample_token_info)
        fetched = await manager.get_session(created.session_id)
        assert fetched.token_info.access_token == sample_token_info.access_token
        assert fetched.token_info.refresh_token == sample_token_info.refresh_token

    async def test_get_session_decrypts_integration_tokens(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """Integration tokens are encrypted at rest and restored for API use."""
        created = await manager.create_session(sample_user, sample_token_info)
        created.integration_tokens["jira"] = {
            "access_token": "jira-private-access-token",
            "refresh_token": "jira-private-refresh-token",
        }
        await manager.update_session(created)

        raw = await manager._get_store().get(RedisKeys.session(created.session_id))
        assert raw is not None
        assert "jira-private-access-token" not in raw
        assert "jira-private-refresh-token" not in raw

        fetched = await manager.get_session(created.session_id, extend_expiry=False)
        assert fetched.integration_tokens["jira"]["access_token"] == "jira-private-access-token"
        assert fetched.integration_tokens["jira"]["refresh_token"] == "jira-private-refresh-token"


class TestUpdateSession:
    """Tests for session update."""

    async def test_update_session_changes_last_accessed(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """update_session advances last_accessed."""
        session = await manager.create_session(sample_user, sample_token_info)
        before = session.last_accessed
        await manager.update_session(session)
        # After update, session.last_accessed is mutated in place
        assert session.last_accessed >= before


class TestRevokeSession:
    """Tests for session revocation."""

    async def test_revoke_session_makes_get_fail(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """After revocation, get_session raises SessionNotFoundError."""
        session = await manager.create_session(sample_user, sample_token_info)
        await manager.revoke_session(session.session_id)
        with pytest.raises(SessionNotFoundError):
            await manager.get_session(session.session_id)

    async def test_revoke_session_removes_session_even_if_file_cleanup_fails(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = await manager.create_session(sample_user, sample_token_info)

        async def fail_file_cleanup(_: str) -> None:
            raise RuntimeError("Storage unavailable")

        monkeypatch.setattr(manager.files, "delete_session_files", fail_file_cleanup)

        await manager.revoke_session(session.session_id)

        with pytest.raises(SessionNotFoundError):
            await manager.get_session(session.session_id)


# =============================================================================
# Fingerprint Validation
# =============================================================================


class TestFingerprintValidation:
    """Tests for session fingerprint binding."""

    async def test_create_with_fingerprint_then_get_same_fingerprint_succeeds(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
        fingerprint_a: SessionFingerprint,
    ) -> None:
        """Creating with fingerprint A and getting with fingerprint A succeeds."""
        session = await manager.create_session(
            sample_user, sample_token_info, fingerprint=fingerprint_a
        )
        fetched = await manager.get_session(
            session.session_id, fingerprint=fingerprint_a
        )
        assert fetched.session_id == session.session_id

    async def test_get_session_fingerprint_mismatch_raises(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
        fingerprint_a: SessionFingerprint,
        fingerprint_b: SessionFingerprint,
    ) -> None:
        """Creating with fingerprint A and getting with fingerprint B raises."""
        session = await manager.create_session(
            sample_user, sample_token_info, fingerprint=fingerprint_a
        )
        with pytest.raises(SessionFingerprintMismatchError):
            await manager.get_session(
                session.session_id, fingerprint=fingerprint_b
            )

    async def test_get_session_no_fingerprint_skips_validation(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
        fingerprint_a: SessionFingerprint,
    ) -> None:
        """Getting a fingerprinted session without providing a fingerprint succeeds."""
        session = await manager.create_session(
            sample_user, sample_token_info, fingerprint=fingerprint_a
        )
        # No fingerprint passed -- validation is skipped
        fetched = await manager.get_session(session.session_id)
        assert fetched.session_id == session.session_id


# =============================================================================
# Absolute Timeout
# =============================================================================


class TestAbsoluteTimeout:
    """Tests for absolute session timeout."""

    async def test_get_session_expired_absolute_timeout(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """A session whose created_at exceeds the absolute timeout is revoked."""
        session = await manager.create_session(sample_user, sample_token_info)

        # Manually overwrite the stored session data with an old created_at
        store = manager._get_store()
        key = RedisKeys.session(session.session_id)
        raw = await store.get(key)
        assert raw is not None

        data = json.loads(raw)
        # Set created_at far enough in the past to exceed absolute timeout
        old_time = (
            datetime.now(timezone.utc) - manager._absolute_timeout - timedelta(hours=1)
        )
        data["created_at"] = old_time.isoformat()
        await store.setex(
            key,
            manager._idle_timeout,
            json.dumps(data, default=str),
        )

        with pytest.raises(SessionExpiredError, match="absolute timeout"):
            await manager.get_session(session.session_id)


# =============================================================================
# User Session Tracking
# =============================================================================


class TestUserSessionTracking:
    """Tests for per-user session tracking."""

    async def test_create_session_tracks_user(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """After creation, the session_id appears in the user's tracked sessions."""
        session = await manager.create_session(sample_user, sample_token_info)
        tracked = await manager.get_user_sessions(sample_user.id)
        assert session.session_id in tracked

    async def test_revoke_session_untracks_user(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """After revoking, the session_id is no longer in the user's tracked sessions."""
        session = await manager.create_session(sample_user, sample_token_info)
        await manager.revoke_session(session.session_id)
        tracked = await manager.get_user_sessions(sample_user.id)
        assert session.session_id not in tracked

    async def test_revoke_all_user_sessions(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """revoke_all_user_sessions removes every session for the user."""
        s1 = await manager.create_session(sample_user, sample_token_info)
        s2 = await manager.create_session(sample_user, sample_token_info)

        count = await manager.revoke_all_user_sessions(sample_user.id)
        assert count == 2

        tracked = await manager.get_user_sessions(sample_user.id)
        assert len(tracked) == 0

        # Both sessions should now be inaccessible
        with pytest.raises(SessionNotFoundError):
            await manager.get_session(s1.session_id)
        with pytest.raises(SessionNotFoundError):
            await manager.get_session(s2.session_id)


# =============================================================================
# Delegation Smoke Tests
# =============================================================================


class TestDelegationDrafts:
    """Smoke tests for draft delegation through the manager."""

    async def test_save_and_get_drafts_via_manager(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
        sample_draft_collection,
    ) -> None:
        """Drafts saved via manager.save_drafts can be retrieved via manager.get_drafts."""
        session = await manager.create_session(sample_user, sample_token_info)
        # Update the draft collection's session_id to match
        sample_draft_collection.session_id = session.session_id
        await manager.save_drafts(session.session_id, sample_draft_collection)
        result = await manager.get_drafts(session.session_id)
        assert result is not None
        assert len(result.items) == len(sample_draft_collection.items)


class TestDelegationConversation:
    """Smoke tests for conversation delegation through the manager."""

    async def test_add_message_via_manager(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """add_message via manager creates a retrievable conversation."""
        session = await manager.create_session(sample_user, sample_token_info)
        conv = await manager.add_message(
            session.session_id, "user", "Hello, world!"
        )
        assert conv is not None
        assert len(conv.messages) >= 1
        assert conv.messages[-1].content == "Hello, world!"


class TestDelegationFileContent:
    """Smoke tests for file content delegation through the manager."""

    async def test_store_and_get_file_content_via_manager(
        self,
        manager: SessionManager,
        sample_user: UserProfile,
        sample_token_info: TokenInfo,
    ) -> None:
        """File content stored via manager is retrievable."""
        session = await manager.create_session(sample_user, sample_token_info)
        await manager.store_file_content(
            session.session_id, "readme.md", "# Hello"
        )
        content = await manager.get_file_content(session.session_id, "readme.md")
        assert content == "# Hello"


class TestDelegationAuthState:
    """Smoke tests for auth state delegation through the manager."""

    async def test_save_and_get_auth_state_via_manager(
        self,
        manager: SessionManager,
    ) -> None:
        """Auth state saved via manager is consumable."""
        state_key = "random-state-value"
        data = {"code_verifier": "abc123", "redirect_uri": "http://localhost"}
        await manager.save_auth_state(state_key, data)
        result = await manager.get_auth_state(state_key)
        assert result is not None
        assert result["code_verifier"] == "abc123"


# =============================================================================
# Health Check
# =============================================================================


class TestHealthCheck:
    """Tests for the storage health check endpoint."""

    async def test_health_check_returns_healthy(
        self, manager: SessionManager
    ) -> None:
        """health_check returns status 'healthy' for an initialized in-memory store."""
        result = await manager.health_check()
        assert result["status"] == "healthy"
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], (int, float))
