"""Session exception hierarchy."""

from __future__ import annotations


class SessionError(Exception):
    """Base exception for session operations."""

    pass


class SessionNotFoundError(SessionError):
    """Session does not exist or has expired."""

    pass


class SessionExpiredError(SessionError):
    """Session has expired (absolute or idle timeout)."""

    pass


class SessionInvalidError(SessionError):
    """Session data is invalid or tampered."""

    pass


class SessionFingerprintMismatchError(SessionError):
    """Session fingerprint doesn't match (potential hijacking)."""

    pass


class RedisConnectionError(SessionError):
    """Redis connection failed."""

    pass
