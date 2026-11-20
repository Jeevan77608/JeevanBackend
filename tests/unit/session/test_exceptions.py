"""Tests for src/session/exceptions.py — session exception hierarchy."""

from __future__ import annotations

import pytest

from src.session.exceptions import (
    RedisConnectionError,
    SessionError,
    SessionExpiredError,
    SessionFingerprintMismatchError,
    SessionInvalidError,
    SessionNotFoundError,
)


class TestSessionError:
    """Tests for the base SessionError."""

    def test_is_base_exception(self) -> None:
        assert issubclass(SessionError, Exception)

    def test_message_propagates(self) -> None:
        err = SessionError("something broke")
        assert str(err) == "something broke"

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(SessionError):
            raise SessionError("test")


class TestSessionNotFoundError:
    """Tests for SessionNotFoundError."""

    def test_inherits_from_session_error(self) -> None:
        assert issubclass(SessionNotFoundError, SessionError)

    def test_message_propagates(self) -> None:
        err = SessionNotFoundError("no such session: abc")
        assert "abc" in str(err)

    def test_caught_by_session_error_handler(self) -> None:
        with pytest.raises(SessionError):
            raise SessionNotFoundError("gone")


class TestSessionExpiredError:
    """Tests for SessionExpiredError."""

    def test_inherits_from_session_error(self) -> None:
        assert issubclass(SessionExpiredError, SessionError)

    def test_message_propagates(self) -> None:
        err = SessionExpiredError("expired after 30m")
        assert "30m" in str(err)


class TestSessionInvalidError:
    """Tests for SessionInvalidError."""

    def test_inherits_from_session_error(self) -> None:
        assert issubclass(SessionInvalidError, SessionError)

    def test_message_propagates(self) -> None:
        err = SessionInvalidError("tampered data")
        assert "tampered" in str(err)


class TestSessionFingerprintMismatchError:
    """Tests for SessionFingerprintMismatchError."""

    def test_inherits_from_session_error(self) -> None:
        assert issubclass(SessionFingerprintMismatchError, SessionError)

    def test_message_propagates(self) -> None:
        err = SessionFingerprintMismatchError("UA hash differs")
        assert "UA hash" in str(err)

    def test_caught_by_session_error_handler(self) -> None:
        with pytest.raises(SessionError):
            raise SessionFingerprintMismatchError("mismatch")


class TestRedisConnectionError:
    """Tests for RedisConnectionError."""

    def test_inherits_from_session_error(self) -> None:
        assert issubclass(RedisConnectionError, SessionError)

    def test_message_propagates(self) -> None:
        err = RedisConnectionError("Connection refused on port 6379")
        assert "6379" in str(err)

    def test_caught_by_session_error_handler(self) -> None:
        with pytest.raises(SessionError):
            raise RedisConnectionError("down")


class TestExceptionHierarchyCompleteness:
    """Cross-cutting tests that verify the full hierarchy."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            SessionNotFoundError,
            SessionExpiredError,
            SessionInvalidError,
            SessionFingerprintMismatchError,
            RedisConnectionError,
        ],
    )
    def test_all_subclasses_inherit_from_session_error(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, SessionError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            SessionNotFoundError,
            SessionExpiredError,
            SessionInvalidError,
            SessionFingerprintMismatchError,
            RedisConnectionError,
        ],
    )
    def test_all_subclasses_can_be_instantiated_without_message(self, exc_cls: type) -> None:
        err = exc_cls()
        assert isinstance(err, SessionError)
