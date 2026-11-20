"""Tests for rate limiting middleware: key functions, config, limiter factory."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from slowapi import Limiter

from src.config.settings import clear_settings_cache
from src.middleware.rate_limit import (
    DEFAULT_RATE_LIMITS,
    RateLimitConfig,
    get_rate_limiter,
    rate_limit_key_func,
    rate_limit_key_ip_only,
    reset_rate_limiter,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_request(
    headers: dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
) -> MagicMock:
    """Build a minimal mock Request for rate-limit key functions.

    Args:
        headers: Optional dict of HTTP headers.
        client_host: IP address to use for request.client.host.

    Returns:
        A MagicMock that behaves like a FastAPI Request.
    """
    request = MagicMock()
    request.headers = headers or {}
    request.client = MagicMock()
    request.client.host = client_host
    # Ensure `hasattr(request.state, "user_id")` returns False by default.
    request.state = MagicMock(spec=[])  # no attributes
    return request


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_limiter() -> Generator[None, None, None]:
    """Reset the rate limiter singleton before and after each test."""
    reset_rate_limiter()
    yield
    reset_rate_limiter()


@pytest.fixture()
def _mock_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set minimal env vars so get_settings() succeeds."""
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret-key-must-be-32-chars!")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("REDIS_ENABLED", "false")
    monkeypatch.setenv("CHAT_SAVE_ENABLED", "false")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("DEFAULT_PUSH_TAG", "@AIGenerated")
    yield
    clear_settings_cache()


# =============================================================================
# RateLimitConfig Tests
# =============================================================================


class TestRateLimitConfig:
    """Tests for the RateLimitConfig frozen dataclass."""

    def test_rate_limit_config_defaults(self) -> None:
        """RateLimitConfig() has the expected default limits."""
        config = RateLimitConfig()
        assert config.standard == "100/minute"
        assert config.tool_execution == "30/minute"
        assert config.auth == "10/minute"
        assert config.health == "200/minute"
        assert config.standard_burst == "10/second"
        assert config.tool_execution_burst == "5/second"

    def test_rate_limit_config_frozen(self) -> None:
        """RateLimitConfig is a frozen dataclass; mutation raises."""
        config = RateLimitConfig()
        with pytest.raises(FrozenInstanceError):
            config.standard = "999/minute"  # type: ignore[misc]

    def test_default_rate_limits_is_config_instance(self) -> None:
        """Module-level DEFAULT_RATE_LIMITS is a RateLimitConfig."""
        assert isinstance(DEFAULT_RATE_LIMITS, RateLimitConfig)


# =============================================================================
# Key Function Tests
# =============================================================================


class TestRateLimitKeyFunc:
    """Tests for rate_limit_key_func — session, user, IP fallback."""

    def test_rate_limit_key_func_uses_session_id(self) -> None:
        """When X-Session-ID header is present, key starts with 'session:'."""
        request = _make_request(headers={"X-Session-ID": "sess-abc-123"})
        key = rate_limit_key_func(request)
        assert key == "session:sess-abc-123"

    def test_rate_limit_key_func_falls_back_to_ip(self) -> None:
        """When no session header and no user_id, key starts with 'ip:'."""
        request = _make_request(client_host="10.0.0.5")
        with patch(
            "src.middleware.rate_limit.get_remote_address",
            return_value="10.0.0.5",
        ):
            key = rate_limit_key_func(request)
        assert key == "ip:10.0.0.5"

    def test_rate_limit_key_func_uses_user_id_from_state(self) -> None:
        """When request.state has user_id, key starts with 'user:'."""
        request = _make_request()
        # Give state a user_id attribute
        request.state = MagicMock()
        request.state.user_id = "user-789"
        key = rate_limit_key_func(request)
        assert key == "user:user-789"


class TestRateLimitKeyIpOnly:
    """Tests for rate_limit_key_ip_only — always IP-based."""

    def test_rate_limit_key_ip_only(self) -> None:
        """rate_limit_key_ip_only always returns an ip-based key."""
        request = _make_request(
            headers={"X-Session-ID": "ignored-session"},
            client_host="192.168.1.1",
        )
        with patch(
            "src.middleware.rate_limit.get_remote_address",
            return_value="192.168.1.1",
        ):
            key = rate_limit_key_ip_only(request)
        assert key == "ip:192.168.1.1"

    def test_rate_limit_key_ip_only_ignores_session(self) -> None:
        """IP-only key ignores X-Session-ID header entirely."""
        request = _make_request(
            headers={"X-Session-ID": "sess-xyz"},
            client_host="172.16.0.1",
        )
        with patch(
            "src.middleware.rate_limit.get_remote_address",
            return_value="172.16.0.1",
        ):
            key = rate_limit_key_ip_only(request)
        assert key.startswith("ip:")
        assert "session:" not in key


# =============================================================================
# Limiter Factory Tests
# =============================================================================


class TestLimiterFactory:
    """Tests for get_rate_limiter / reset_rate_limiter singleton."""

    def test_get_rate_limiter_returns_limiter(self, _mock_env: None) -> None:
        """get_rate_limiter() returns a Limiter instance."""
        limiter = get_rate_limiter()
        assert isinstance(limiter, Limiter)

    def test_get_rate_limiter_is_singleton(self, _mock_env: None) -> None:
        """Calling get_rate_limiter() twice returns the same instance."""
        first = get_rate_limiter()
        second = get_rate_limiter()
        assert first is second

    def test_reset_rate_limiter(self, _mock_env: None) -> None:
        """reset_rate_limiter() clears the singleton; next call creates new."""
        first = get_rate_limiter()
        reset_rate_limiter()
        second = get_rate_limiter()
        assert first is not second
