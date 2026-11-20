"""Shared fixtures and early environment setup for integration tests.

Environment variables are set at import time so that any module-level code
(e.g. ``get_rate_limiter()`` called at ``from src.api.routes import router``)
sees valid settings and does not raise ``ValidationError``.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

import httpx
import pytest

# ---------------------------------------------------------------------------
# Early env-var bootstrap -- MUST run before any ``src.*`` import
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "APP_SECRET_KEY": "test-secret-key-must-be-32-chars!",
    "AUTH_ENABLED": "false",
    "REDIS_ENABLED": "false",
    "CHAT_SAVE_ENABLED": "false",
    "APP_ENV": "development",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "test-key",
    "DEFAULT_PUSH_TAG": "@AIGenerated",
    "CORS_ORIGINS": "http://localhost:3000",
    "ENCRYPTION_KEY": "dGVzdC1lbmNyeXB0aW9uLWtleS0xMjM0NTY3ODkwYWI=",
}

for key, value in _DEFAULTS.items():
    os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Shared app_client fixture — triggers lifespan manually
# ---------------------------------------------------------------------------

from src.config.settings import Settings, clear_settings_cache
from src.middleware.rate_limit import reset_rate_limiter


@pytest.fixture()
async def app_client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a test app with httpx client and manually initialize app state.

    httpx.ASGITransport does NOT trigger the FastAPI lifespan, so we
    manually create and attach the SessionManager to app.state.

    Yields:
        An ``httpx.AsyncClient`` bound to the ASGI application under test.
    """
    for key, value in _DEFAULTS.items():
        monkeypatch.setenv(key, value)

    clear_settings_cache()
    reset_rate_limiter()

    from src.main import create_app
    from src.session.manager import SessionManager

    app = create_app()

    # Manually initialize session manager (lifespan won't run with ASGITransport)
    settings = Settings()
    session_manager = SessionManager(settings)
    await session_manager.initialize()
    app.state.session_manager = session_manager
    app.state.oauth_provider = None  # AUTH_ENABLED=false

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    await session_manager.close()
    clear_settings_cache()
    reset_rate_limiter()
