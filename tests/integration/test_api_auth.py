"""Integration tests for /api/v1/auth/start and /api/v1/auth/callback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.config.settings import Settings, clear_settings_cache
from src.middleware.rate_limit import reset_rate_limiter
from src.session.manager import SessionManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
async def auth_app_client(monkeypatch):
    """Create an ASGI test client with AUTH_ENABLED=true and Azure AD env vars.

    The fixture stands up a fresh FastAPI app via ``create_app()`` with
    authentication enabled and Azure AD env vars populated.  Because
    ``httpx.ASGITransport`` does not trigger FastAPI lifespan events,
    ``app.state`` is manually initialised with a ``SessionManager`` and
    a mocked ``AzureADOAuthProvider``.

    Yields:
        Tuple of (httpx.AsyncClient, FastAPI app).
    """
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret-key-must-be-32-chars!")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("REDIS_ENABLED", "false")
    monkeypatch.setenv("CHAT_SAVE_ENABLED", "false")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_AD_TENANT_ID", "a" * 36)
    monkeypatch.setenv("AZURE_AD_CLIENT_ID", "b" * 36)
    monkeypatch.setenv("AZURE_AD_CLIENT_SECRET", "secret")
    monkeypatch.setenv("AZURE_AD_REDIRECT_URI", "https://example.com/api/auth/callback")
    monkeypatch.setenv("DEFAULT_PUSH_TAG", "@AIGenerated")
    monkeypatch.setenv(
        "ENCRYPTION_KEY",
        "dGVzdC1lbmNyeXB0aW9uLWtleS0xMjM0NTY3ODkwYWI=",
    )

    clear_settings_cache()
    reset_rate_limiter()

    with patch("src.auth.oauth.ConfidentialClientApplication"):
        from src.main import create_app

        app = create_app()

        # Manually initialise app.state (lifespan does not run with ASGITransport)
        settings = Settings()
        session_manager = SessionManager(settings)
        await session_manager.initialize()
        app.state.session_manager = session_manager

        from src.auth.oauth import AzureADOAuthProvider

        oauth_provider = AzureADOAuthProvider(settings)
        app.state.oauth_provider = oauth_provider

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, app

    await session_manager.close()
    clear_settings_cache()
    reset_rate_limiter()


@pytest.fixture()
async def noauth_app_client(monkeypatch):
    """Create an ASGI test client with AUTH_ENABLED=false.

    Useful for asserting that the OAuth provider is ``None`` when
    authentication is disabled, causing auth endpoints to return 503.

    Yields:
        Tuple of (httpx.AsyncClient, FastAPI app).
    """
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret-key-must-be-32-chars!")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("REDIS_ENABLED", "false")
    monkeypatch.setenv("CHAT_SAVE_ENABLED", "false")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DEFAULT_PUSH_TAG", "@AIGenerated")
    monkeypatch.setenv(
        "ENCRYPTION_KEY",
        "dGVzdC1lbmNyeXB0aW9uLWtleS0xMjM0NTY3ODkwYWI=",
    )

    clear_settings_cache()
    reset_rate_limiter()

    from src.main import create_app

    app = create_app()

    # Manually initialise app.state
    settings = Settings()
    session_manager = SessionManager(settings)
    await session_manager.initialize()
    app.state.session_manager = session_manager
    app.state.oauth_provider = None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, app

    await session_manager.close()
    clear_settings_cache()
    reset_rate_limiter()


# =============================================================================
# Tests -- /api/v1/auth/start
# =============================================================================


async def test_auth_start_returns_authorization_url(auth_app_client) -> None:
    """POST /api/v1/auth/start with valid redirect_uri returns authorization_url and state."""
    client, app = auth_app_client

    # Mock the OAuth provider's methods on the already-initialised app.state
    mock_auth_state = MagicMock()
    mock_auth_state.state = "test-state-value"
    mock_auth_state.pkce = MagicMock()
    mock_auth_state.pkce.code_verifier = "v" * 64
    mock_auth_state.pkce.code_challenge = "c" * 43
    mock_auth_state.redirect_uri = "https://example.com/api/auth/callback"
    mock_auth_state.to_dict.return_value = {
        "state": "test-state-value",
        "code_verifier": "v" * 64,
        "code_challenge": "c" * 43,
        "redirect_uri": "https://example.com/api/auth/callback",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    provider = app.state.oauth_provider
    provider.create_authorization_state = MagicMock(return_value=mock_auth_state)
    provider.get_authorization_url = MagicMock(
        return_value="https://login.microsoftonline.com/authorize?state=test-state-value"
    )

    response = await client.post(
        "/api/v1/auth/start",
        json={"redirect_uri": "https://example.com/api/auth/callback"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "authorization_url" in body
    assert "state" in body
    assert body["state"] == "test-state-value"
    assert body["authorization_url"].startswith("https://login.microsoftonline.com")


async def test_auth_start_invalid_redirect_uri_returns_400(auth_app_client) -> None:
    """POST /api/v1/auth/start with a redirect_uri that doesn't match config returns 400."""
    client, _app = auth_app_client

    response = await client.post(
        "/api/v1/auth/start",
        json={"redirect_uri": "https://evil.com/callback"},
    )

    assert response.status_code == 400
    assert "Invalid redirect URI" in response.json()["detail"]


async def test_auth_start_no_oauth_provider_returns_503(auth_app_client) -> None:
    """When app.state.oauth_provider is forcibly set to None, start returns 503."""
    client, app = auth_app_client

    # Forcibly remove the provider to simulate it being unavailable
    original_provider = app.state.oauth_provider
    app.state.oauth_provider = None

    try:
        response = await client.post(
            "/api/v1/auth/start",
            json={"redirect_uri": "https://example.com/api/auth/callback"},
        )

        assert response.status_code == 503
        assert "OAuth provider not available" in response.json()["detail"]
    finally:
        app.state.oauth_provider = original_provider


# =============================================================================
# Tests -- /api/v1/auth/callback
# =============================================================================


async def test_auth_callback_invalid_state_returns_400(auth_app_client) -> None:
    """POST /api/v1/auth/callback with an unknown state returns 400."""
    client, _app = auth_app_client

    response = await client.post(
        "/api/v1/auth/callback",
        json={"code": "some-auth-code", "state": "nonexistent-state-value"},
    )

    # The session manager will not find the auth state, so 400 is expected
    assert response.status_code == 400
    body = response.json()
    assert "Invalid or expired state" in body["detail"]


async def test_auth_callback_no_oauth_provider_returns_503(auth_app_client) -> None:
    """When app.state.oauth_provider is None, callback returns 503."""
    client, app = auth_app_client

    original_provider = app.state.oauth_provider
    app.state.oauth_provider = None

    try:
        response = await client.post(
            "/api/v1/auth/callback",
            json={"code": "some-auth-code", "state": "some-state"},
        )

        assert response.status_code == 503
        assert "OAuth provider not available" in response.json()["detail"]
    finally:
        app.state.oauth_provider = original_provider


# =============================================================================
# Tests -- AUTH_ENABLED=false
# =============================================================================


async def test_auth_disabled_start_returns_no_provider(noauth_app_client) -> None:
    """When AUTH_ENABLED=false, oauth_provider is None and /auth/start returns 503.

    The ``get_oauth_provider`` dependency returns ``None`` when auth is
    disabled, and the route handler raises 503 for a ``None`` provider.
    """
    client, app = noauth_app_client

    # Verify provider is None on startup
    assert app.state.oauth_provider is None

    response = await client.post(
        "/api/v1/auth/start",
        json={"redirect_uri": "https://example.com/api/auth/callback"},
    )

    assert response.status_code == 503
    body = response.json()
    assert "OAuth provider not available" in body["detail"]
