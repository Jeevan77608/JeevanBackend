"""Integration tests for application startup, shutdown, middleware, and CORS."""

from __future__ import annotations

import pytest
import httpx

from src.config.settings import clear_settings_cache
from src.middleware.rate_limit import reset_rate_limiter


# Uses shared app_client fixture from tests/integration/conftest.py


# =============================================================================
# TestAppCreation
# =============================================================================


class TestAppCreation:
    """Tests that verify the application factory and basic metadata."""

    async def test_app_creates_successfully(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``create_app()`` returns a FastAPI instance without errors."""
        monkeypatch.setenv("APP_SECRET_KEY", "test-secret-key-must-be-32-chars!")
        monkeypatch.setenv("AUTH_ENABLED", "false")
        monkeypatch.setenv("REDIS_ENABLED", "false")
        monkeypatch.setenv("CHAT_SAVE_ENABLED", "false")
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("DEFAULT_PUSH_TAG", "@AIGenerated")
        monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
        monkeypatch.setenv(
            "ENCRYPTION_KEY",
            "dGVzdC1lbmNyeXB0aW9uLWtleS0xMjM0NTY3ODkwYWI=",
        )

        clear_settings_cache()
        reset_rate_limiter()

        from fastapi import FastAPI
        from src.main import create_app

        app = create_app()
        assert isinstance(app, FastAPI)

        clear_settings_cache()
        reset_rate_limiter()

    async def test_root_endpoint_returns_api_info(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """GET / returns the API name and version."""
        response = await app_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Azure DevOps MCP Server"
        assert data["version"] == "1.0.0"
        assert "health" in data


# =============================================================================
# TestHealthAndConfig
# =============================================================================


class TestHealthAndConfig:
    """Tests for health check and configuration endpoints."""

    async def test_health_endpoint_exists(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """GET /api/v1/health returns 200 with a ``status`` field."""
        response = await app_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    async def test_config_endpoint_returns_feature_flags(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """GET /api/v1/config returns ``chat_save_enabled`` and ``default_push_tag``."""
        response = await app_client.get("/api/v1/config")
        assert response.status_code == 200

        data = response.json()
        assert "chat_save_enabled" in data
        assert data["chat_save_enabled"] is False
        assert "default_push_tag" in data
        assert data["default_push_tag"] == "@AIGenerated"


# =============================================================================
# TestCORS
# =============================================================================


class TestCORS:
    """Tests that verify CORS middleware behaviour."""

    async def test_cors_headers_present_for_allowed_origin(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """An OPTIONS pre-flight from an allowed origin receives CORS headers."""
        response = await app_client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        # FastAPI/Starlette CORS middleware responds 200 to pre-flight
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


# =============================================================================
# TestApplicationState
# =============================================================================


class TestApplicationState:
    """Tests that verify objects created during the lifespan are on app.state."""

    async def test_session_manager_on_app_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After startup, ``app.state.session_manager`` exists."""
        monkeypatch.setenv("APP_SECRET_KEY", "test-secret-key-must-be-32-chars!")
        monkeypatch.setenv("AUTH_ENABLED", "false")
        monkeypatch.setenv("REDIS_ENABLED", "false")
        monkeypatch.setenv("CHAT_SAVE_ENABLED", "false")
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("DEFAULT_PUSH_TAG", "@AIGenerated")
        monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
        monkeypatch.setenv(
            "ENCRYPTION_KEY",
            "dGVzdC1lbmNyeXB0aW9uLWtleS0xMjM0NTY3ODkwYWI=",
        )

        clear_settings_cache()
        reset_rate_limiter()

        from src.main import create_app, lifespan
        from src.session.manager import SessionManager

        app = create_app()

        async with lifespan(app):
            assert hasattr(app.state, "session_manager")
            assert isinstance(app.state.session_manager, SessionManager)

        clear_settings_cache()
        reset_rate_limiter()

    async def test_rate_limiter_on_app_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After startup, ``app.state.limiter`` exists."""
        monkeypatch.setenv("APP_SECRET_KEY", "test-secret-key-must-be-32-chars!")
        monkeypatch.setenv("AUTH_ENABLED", "false")
        monkeypatch.setenv("REDIS_ENABLED", "false")
        monkeypatch.setenv("CHAT_SAVE_ENABLED", "false")
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("DEFAULT_PUSH_TAG", "@AIGenerated")
        monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
        monkeypatch.setenv(
            "ENCRYPTION_KEY",
            "dGVzdC1lbmNyeXB0aW9uLWtleS0xMjM0NTY3ODkwYWI=",
        )

        clear_settings_cache()
        reset_rate_limiter()

        from slowapi import Limiter
        from src.main import create_app, lifespan

        app = create_app()

        async with lifespan(app):
            assert hasattr(app.state, "limiter")
            assert isinstance(app.state.limiter, Limiter)

        clear_settings_cache()
        reset_rate_limiter()


# =============================================================================
# TestOpenAPI
# =============================================================================


class TestOpenAPI:
    """Tests for OpenAPI schema availability."""

    async def test_openapi_available_in_development(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """/openapi.json returns 200 when APP_ENV is development."""
        response = await app_client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert data["info"]["title"] == "Azure DevOps MCP Server"
