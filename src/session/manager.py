"""
Session management following OWASP guidelines.

This module implements industry-standard session management with:
- Secure session ID generation (256 bits of entropy)
- Encrypted token storage (AES-256-GCM)
- Session binding via fingerprinting
- Sliding window expiration with absolute timeout
- Redis or in-memory storage (configurable via REDIS_ENABLED)
- Comprehensive audit logging

References:
- OWASP Session Management Cheat Sheet
- NIST SP 800-63B (Digital Identity Guidelines)
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from src.models.devops import BoardContext

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import Settings
from src.models.user import SessionContext, TokenInfo, UserProfile
from src.models.work_items import DraftCollection, DraftWorkItem
from src.session.encryption import DecryptionError, EncryptionService
from src.session.memory_store import InMemoryStore
from src.session.models import (
    ConversationHistory,
    SessionFingerprint,
    SessionState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions (re-exported for backward compatibility)
# =============================================================================

from src.session.exceptions import (  # noqa: E402, F811
    RedisConnectionError,
    SessionError,
    SessionExpiredError,
    SessionFingerprintMismatchError,
    SessionInvalidError,
    SessionNotFoundError,
)

# =============================================================================
# Constants
# =============================================================================
from src.session.redis_keys import RedisKeys  # noqa: E402, F811

# Session ID entropy: 256 bits = 32 bytes (OWASP recommends minimum 128 bits)
SESSION_ID_BYTES: Final[int] = 32


# =============================================================================
# Repository imports
# =============================================================================

from src.session.auth_state_store import AuthStateStore  # noqa: E402
from src.session.board_context_store import BoardContextStore  # noqa: E402
from src.session.conversation_store import ConversationStore  # noqa: E402
from src.session.draft_repository import DraftRepository  # noqa: E402
from src.session.file_content_store import FileContentStore  # noqa: E402

# =============================================================================
# Session Manager
# =============================================================================


class SessionManager:
    """
    Enterprise-grade session manager with Redis or in-memory backend.

    Composes specialized repositories for different concerns while
    maintaining backward-compatible delegation methods.

    Features:
    - Secure session ID generation (256-bit entropy)
    - AES-256-GCM encrypted token storage
    - Optional session fingerprinting
    - Sliding window + absolute timeout
    - Redis or in-memory storage (configurable)
    - Comprehensive audit logging

    Thread Safety:
    - This class is thread-safe for async operations.
    - Redis operations are atomic where needed.
    """

    # Type alias for storage backends
    StoreType = redis.Redis | InMemoryStore

    def __init__(
        self,
        settings: Settings,
        *,
        enable_fingerprinting: bool = True,
        strict_ip_binding: bool = False,
    ) -> None:
        self._settings = settings
        self._use_redis = settings.redis_enabled
        self._pool: ConnectionPool | None = None
        self._redis: redis.Redis | None = None
        self._memory_store: InMemoryStore | None = None
        self._encryption = EncryptionService(
            settings.encryption_key.get_secret_value()
        )

        # Timeouts
        self._idle_timeout = timedelta(hours=settings.session_expiry_hours)
        self._absolute_timeout = timedelta(hours=settings.session_expiry_hours * 2)

        # Security settings
        self._enable_fingerprinting = enable_fingerprinting
        self._strict_ip_binding = strict_ip_binding

        # Repositories (initialized in initialize())
        self._drafts: DraftRepository | None = None
        self._conversations: ConversationStore | None = None
        self._files: FileContentStore | None = None
        self._board_context: BoardContextStore | None = None
        self._auth_state: AuthStateStore | None = None
        self._file_cleanup_task: asyncio.Task[None] | None = None

        logger.info(
            "Session manager configured",
            extra={
                "idle_timeout_hours": settings.session_expiry_hours,
                "absolute_timeout_hours": settings.session_expiry_hours * 2,
                "fingerprinting_enabled": enable_fingerprinting,
                "storage_backend": "redis" if self._use_redis else "in-memory",
            },
        )

    # =========================================================================
    # Repository Properties
    # =========================================================================

    @property
    def drafts(self) -> DraftRepository:
        """Draft work items repository."""
        assert self._drafts is not None, "SessionManager not initialized"
        return self._drafts

    @property
    def conversations(self) -> ConversationStore:
        """Conversation history repository."""
        assert self._conversations is not None, "SessionManager not initialized"
        return self._conversations

    @property
    def files(self) -> FileContentStore:
        """File content repository."""
        assert self._files is not None, "SessionManager not initialized"
        return self._files

    @property
    def board_context_store(self) -> BoardContextStore:
        """Board context repository."""
        assert self._board_context is not None, "SessionManager not initialized"
        return self._board_context

    @property
    def auth_state_store(self) -> AuthStateStore:
        """Auth state repository."""
        assert self._auth_state is not None, "SessionManager not initialized"
        return self._auth_state

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize storage backend and repositories."""
        if self._use_redis:
            await self._initialize_redis()
        else:
            await self._initialize_memory_store()

        # Initialize repositories with the store
        store = self._get_store()
        idle = self._idle_timeout
        self._drafts = DraftRepository(store, idle)
        self._conversations = ConversationStore(store, idle, self._drafts)
        self._files = FileContentStore(store, idle, self._settings)
        self._board_context = BoardContextStore(store, idle)
        self._auth_state = AuthStateStore(store, idle)
        if self._settings.supabase_storage_enabled:
            await self._files.cleanup_expired_files()
            self._file_cleanup_task = asyncio.create_task(
                self._cleanup_expired_files()
            )

    async def _cleanup_expired_files(self) -> None:
        """Periodically remove expired private Supabase Storage objects."""
        while True:
            await asyncio.sleep(3600)
            if self._files:
                try:
                    await self._files.cleanup_expired_files()
                except Exception:
                    logger.exception("Could not clean expired Supabase Storage files")

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection pool."""
        redis_url = self._settings.redis.url
        password = (
            self._settings.redis.password.get_secret_value()
            if self._settings.redis.password
            else None
        )

        try:
            self._pool = ConnectionPool.from_url(
                redis_url,
                password=password,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self._settings.redis.max_connections,
                socket_timeout=self._settings.redis.socket_timeout,
                socket_connect_timeout=self._settings.redis.connect_timeout,
                retry_on_timeout=True,
            )

            self._redis = redis.Redis(connection_pool=self._pool)

            # Test connection
            await self._redis.ping()
            logger.info("Redis connection pool established")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise RedisConnectionError(f"Redis connection failed: {e}") from e

    async def _initialize_memory_store(self) -> None:
        """Initialize in-memory storage (development only)."""
        logger.warning(
            "Using in-memory session storage. "
            "Sessions will be lost on restart. "
            "This should only be used for local development!"
        )
        self._memory_store = InMemoryStore()
        await self._memory_store.initialize()

    async def close(self) -> None:
        """Close storage backend gracefully."""
        if self._file_cleanup_task:
            self._file_cleanup_task.cancel()
            try:
                await self._file_cleanup_task
            except asyncio.CancelledError:
                pass
        if self._use_redis:
            if self._redis:
                await self._redis.close()
            if self._pool:
                await self._pool.disconnect()
            logger.info("Redis connection pool closed")
        else:
            if self._memory_store:
                await self._memory_store.close()

    def _get_store(self) -> StoreType:
        """Get storage backend, ensuring it's initialized."""
        if self._use_redis:
            if self._redis is None:
                raise RedisConnectionError("Session manager not initialized")
            return self._redis
        else:
            if self._memory_store is None:
                raise RedisConnectionError("Session manager not initialized")
            return self._memory_store

    # =========================================================================
    # Session ID Generation
    # =========================================================================

    @staticmethod
    def generate_session_id() -> str:
        """Generate cryptographically secure session ID (256-bit entropy)."""
        return secrets.token_urlsafe(SESSION_ID_BYTES)

    @staticmethod
    def generate_state() -> str:
        """Generate OAuth state parameter."""
        return secrets.token_urlsafe(32)

    # =========================================================================
    # Session CRUD with Retry Logic
    # =========================================================================

    @retry(
        retry=retry_if_exception_type(redis.RedisError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    )
    async def create_session(
        self,
        user: UserProfile,
        token_info: TokenInfo | None = None,
        fingerprint: SessionFingerprint | None = None,
    ) -> SessionContext:
        """Create a new session for authenticated user."""
        session_id = self.generate_session_id()
        now = datetime.now(timezone.utc)

        session = SessionContext(
            session_id=session_id,
            user=user,
            token_info=token_info,
            created_at=now,
            last_accessed=now,
        )

        await self._save_session_internal(session, fingerprint)
        await self._track_user_session(user.id, session_id)

        logger.info(
            "Session created",
            extra={
                "session_id": session_id[:16] + "...",
                "user_id": user.id,
            },
        )

        return session

    async def _save_session_internal(
        self,
        session: SessionContext,
        fingerprint: SessionFingerprint | None = None,
    ) -> None:
        """Internal method to save session with encryption."""
        r = self._get_store()
        key = RedisKeys.session(session.session_id)

        session_data = session.model_dump(mode="json")

        # Encrypt sensitive token data with session binding (AAD)
        if session_data["token_info"] is not None:
            session_data["token_info"] = self._encryption.encrypt_dict(
                session_data["token_info"],
                associated_data=session.session_id.encode(),
            )
        encrypted_integrations = self._encryption.encrypt_dict(
            session_data["integration_tokens"],
            associated_data=f"{session.session_id}:integrations".encode(),
        )
        session_data["integration_tokens"] = encrypted_integrations

        if fingerprint:
            session_data["fingerprint"] = fingerprint.model_dump()

        session_data["_state"] = SessionState.ACTIVE.value
        session_data["_version"] = 1

        await r.setex(
            key,
            self._idle_timeout,
            json.dumps(session_data, default=str),
        )

    @retry(
        retry=retry_if_exception_type(redis.RedisError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    )
    async def get_session(
        self,
        session_id: str,
        fingerprint: SessionFingerprint | None = None,
        *,
        extend_expiry: bool = True,
    ) -> SessionContext:
        """Retrieve and validate session."""
        r = self._get_store()
        key = RedisKeys.session(session_id)

        data = await r.get(key)
        if not data:
            raise SessionNotFoundError(f"Session not found: {session_id[:16]}...")

        try:
            session_data = json.loads(data)

            # Check state
            state = session_data.get("_state", SessionState.ACTIVE.value)
            if state != SessionState.ACTIVE.value:
                raise SessionExpiredError(f"Session is {state}")

            # Check absolute timeout
            created_at = datetime.fromisoformat(
                session_data["created_at"].replace("Z", "+00:00")
            )
            if datetime.now(timezone.utc) - created_at > self._absolute_timeout:
                await self.revoke_session(session_id)
                raise SessionExpiredError("Session exceeded absolute timeout")

            # Validate fingerprint if enabled
            if self._enable_fingerprinting and fingerprint:
                stored_fp_data = session_data.get("fingerprint")
                if stored_fp_data:
                    stored_fp = SessionFingerprint.model_validate(stored_fp_data)
                    if not stored_fp.matches(
                        fingerprint, strict_ip=self._strict_ip_binding
                    ):
                        logger.warning(
                            "Session fingerprint mismatch",
                            extra={"session_id": session_id[:16] + "..."},
                        )
                        raise SessionFingerprintMismatchError(
                            "Session fingerprint mismatch"
                        )

            # Decrypt token data
            encrypted_token = session_data.get("token_info")
            if isinstance(encrypted_token, str):
                session_data["token_info"] = self._encryption.decrypt_dict(
                    encrypted_token,
                    associated_data=session_id.encode(),
                )
            encrypted_integrations = session_data.get("integration_tokens")
            if isinstance(encrypted_integrations, str):
                session_data["integration_tokens"] = self._encryption.decrypt_dict(
                    encrypted_integrations,
                    associated_data=f"{session_id}:integrations".encode(),
                )

            # Remove internal fields before validation
            session_data.pop("_state", None)
            session_data.pop("_version", None)
            session_data.pop("fingerprint", None)

            session = SessionContext.model_validate(session_data)

            # Extend idle timeout (sliding window)
            if extend_expiry:
                session.last_accessed = datetime.now(timezone.utc)
                await self._save_session_internal(session, fingerprint)

            return session

        except (SessionError, DecryptionError):
            raise
        except Exception as e:
            logger.error(
                f"Failed to parse session data: {e}",
                extra={"session_id": session_id[:16] + "..."},
            )
            raise SessionInvalidError(f"Invalid session data: {e}") from e

    async def update_session(
        self,
        session: SessionContext,
        fingerprint: SessionFingerprint | None = None,
    ) -> None:
        """Update an existing session."""
        session.last_accessed = datetime.now(timezone.utc)
        await self._save_session_internal(session, fingerprint)

    async def revoke_session(self, session_id: str) -> None:
        """Revoke a session (logout) and delete associated data."""
        r = self._get_store()

        try:
            session = await self.get_session(session_id, extend_expiry=False)
            await self._untrack_user_session(session.user.id, session_id)
        except SessionError:
            pass

        keys = [
            RedisKeys.session(session_id),
            RedisKeys.drafts(session_id),
            RedisKeys.conversation(session_id),
        ]
        if self._files:
            try:
                await self._files.delete_session_files(session_id)
            except Exception:
                logger.exception(
                    "Could not delete private file context for revoked session",
                    extra={"session_id": session_id[:16] + "..."},
                )
        await r.delete(*keys)

        logger.info(
            "Session revoked",
            extra={"session_id": session_id[:16] + "..."},
        )

    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user (force logout from all devices)."""
        r = self._get_store()
        key = RedisKeys.user_sessions(user_id)

        session_ids = await r.smembers(key)
        count = 0

        for session_id in session_ids:
            try:
                await self.revoke_session(session_id)
                count += 1
            except SessionError:
                pass

        await r.delete(key)
        logger.info(f"Revoked {count} sessions for user {user_id}")
        return count

    # =========================================================================
    # User Session Tracking
    # =========================================================================

    async def _track_user_session(self, user_id: str, session_id: str) -> None:
        """Track session ID for a user (multi-device support)."""
        r = self._get_store()
        key = RedisKeys.user_sessions(user_id)
        await r.sadd(key, session_id)
        await r.expire(key, self._absolute_timeout)

    async def _untrack_user_session(self, user_id: str, session_id: str) -> None:
        """Remove session from user's tracked sessions."""
        r = self._get_store()
        key = RedisKeys.user_sessions(user_id)
        await r.srem(key, session_id)

    async def get_user_sessions(self, user_id: str) -> list[str]:
        """Get all active session IDs for a user."""
        r = self._get_store()
        key = RedisKeys.user_sessions(user_id)
        return list(await r.smembers(key))

    # =========================================================================
    # Delegation: Draft Work Items
    # =========================================================================

    async def save_drafts(self, session_id: str, drafts: DraftCollection) -> None:
        """Save draft work items collection for a session."""
        return await self.drafts.save_drafts(session_id, drafts)

    async def get_drafts(self, session_id: str) -> DraftCollection | None:
        """Retrieve draft collection for a session."""
        return await self.drafts.get_drafts(session_id)

    async def delete_drafts(self, session_id: str) -> None:
        """Delete draft collection for a session."""
        return await self.drafts.delete_drafts(session_id)

    async def replace_drafts_from_snapshot(
        self,
        session_id: str,
        drafts_snapshot: list[dict[str, Any]],
        project_id: str,
    ) -> None:
        """Replace session drafts with the given snapshot."""
        return await self.drafts.replace_drafts_from_snapshot(
            session_id, drafts_snapshot, project_id
        )

    async def update_draft_item(
        self, session_id: str, draft_item: DraftWorkItem
    ) -> DraftCollection:
        """Update a single draft item in the collection."""
        return await self.drafts.update_draft_item(session_id, draft_item)

    async def get_draft(self, session_id: str, draft_id: str) -> DraftWorkItem | None:
        """Retrieve a single draft item by ID."""
        return await self.drafts.get_draft(session_id, draft_id)

    async def delete_draft(self, session_id: str, draft_id: str) -> bool:
        """Delete a single draft item by ID."""
        return await self.drafts.delete_draft(session_id, draft_id)

    # =========================================================================
    # Delegation: Conversation History
    # =========================================================================

    async def save_conversation(
        self, session_id: str, conversation: ConversationHistory
    ) -> None:
        """Save conversation history for a session."""
        return await self.conversations.save_conversation(session_id, conversation)

    async def get_conversation(
        self, session_id: str
    ) -> ConversationHistory | None:
        """Retrieve conversation history for a session."""
        return await self.conversations.get_conversation(session_id)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationHistory:
        """Add a message to conversation history."""
        return await self.conversations.add_message(
            session_id, role, content, metadata
        )

    async def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history (keeps system messages)."""
        return await self.conversations.clear_conversation(session_id)

    async def save_draft_conversation(
        self,
        session_id: str,
        messages_snapshot: list[dict[str, Any]],
        title: str = "",
    ) -> None:
        """Store draft conversation for this session."""
        return await self.conversations.save_draft_conversation(
            session_id, messages_snapshot, title
        )

    async def get_draft_conversation(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Return draft conversation for this session, or None."""
        return await self.conversations.get_draft_conversation(session_id)

    async def clear_draft_conversation(self, session_id: str) -> None:
        """Remove draft conversation and clear draft work items."""
        return await self.conversations.clear_draft_conversation(session_id)

    # =========================================================================
    # Delegation: File Content
    # =========================================================================

    async def store_file_content(
        self, session_id: str, file_id: str, content: str
    ) -> None:
        """Store file content for use in work item generation."""
        return await self.files.store_file_content(session_id, file_id, content)

    async def get_file_content(
        self, session_id: str, file_id: str
    ) -> str | None:
        """Retrieve stored file content."""
        return await self.files.get_file_content(session_id, file_id)

    async def list_file_contents(self, session_id: str) -> list[str]:
        """List all file IDs stored for a session."""
        return await self.files.list_file_contents(session_id)

    # =========================================================================
    # Delegation: Board Context
    # =========================================================================

    async def store_board_context(
        self,
        session_id: str,
        project_id: str,
        board_context: BoardContext,
    ) -> None:
        """Store BoardContext for a project in the session."""
        return await self.board_context_store.store_board_context(
            session_id, project_id, board_context
        )

    async def get_board_context(
        self,
        session_id: str,
        project_id: str,
    ) -> BoardContext | None:
        """Retrieve stored BoardContext for a project."""
        return await self.board_context_store.get_board_context(
            session_id, project_id
        )

    # =========================================================================
    # Delegation: Auth State
    # =========================================================================

    async def save_auth_state(
        self,
        state: str,
        data: dict[str, Any],
        ttl_seconds: int = 600,
    ) -> None:
        """Save OAuth state for CSRF protection."""
        return await self.auth_state_store.save_auth_state(state, data, ttl_seconds)

    async def get_auth_state(self, state: str) -> dict[str, Any] | None:
        """Retrieve and consume OAuth state (one-time use)."""
        return await self.auth_state_store.get_auth_state(state)

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check storage backend health."""
        try:
            r = self._get_store()
            start = datetime.now(timezone.utc)
            await r.ping()
            latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            info = await r.info(section="memory")

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", "unknown"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
