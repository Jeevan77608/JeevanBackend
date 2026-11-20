from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

# Importing the API router initializes its rate limiter, which validates settings.
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-api-key")
os.environ.setdefault("APP_SECRET_KEY", "test-secret-key-must-be-32-chars!")

from src.api import routes


class SupabaseAuthResponse:
    status_code = 200
    is_success = True

    @staticmethod
    def json() -> dict[str, object]:
        return {
            "id": "8bd04cb8-8ee8-40b6-bdb3-22b738920730",
            "email": "jee@example.com",
            "user_metadata": {"full_name": "Jeevan"},
        }


class InvalidSupabaseAuthResponse:
    status_code = 401
    is_success = False


class SupabaseAuthHttpClient:
    async def __aenter__(self) -> SupabaseAuthHttpClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get(self, url: str, *, headers: dict[str, str]) -> SupabaseAuthResponse:
        assert url == "https://example.supabase.co/auth/v1/user"
        assert headers["Authorization"] == "Bearer valid-token"
        assert headers["apikey"] == "anon-key"
        return SupabaseAuthResponse()


class InvalidSupabaseAuthHttpClient:
    async def __aenter__(self) -> InvalidSupabaseAuthHttpClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get(self, url: str, *, headers: dict[str, str]) -> InvalidSupabaseAuthResponse:
        assert url == "https://example.supabase.co/auth/v1/user"
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["apikey"] == "anon-key"
        return InvalidSupabaseAuthResponse()


def test_supabase_access_token_creates_encrypted_app_session(
    monkeypatch,
) -> None:
    settings = SimpleNamespace(
        auth_enabled=True,
        auth_provider="supabase",
        supabase_url="https://example.supabase.co",
        supabase_anon_key=SecretStr("anon-key"),
        session_expiry_hours=24,
    )
    monkeypatch.setattr(routes, "get_settings", lambda: settings)

    def create_http_client(*, timeout: float) -> SupabaseAuthHttpClient:
        assert timeout > 0
        return SupabaseAuthHttpClient()

    monkeypatch.setattr(
        routes.httpx,
        "AsyncClient",
        create_http_client,
    )

    session = SimpleNamespace(
        session_id="app-session-id",
        metadata={},
    )
    manager = SimpleNamespace(
        create_session=AsyncMock(return_value=session),
        update_session=AsyncMock(),
    )
    app = FastAPI()
    app.state.limiter = routes.limiter
    app.state.session_manager = manager
    app.include_router(routes.router)

    with TestClient(app) as client:
        response = client.post(
            "/auth/supabase/session",
            headers={"Authorization": "Bearer valid-token"},
        )

    assert response.status_code == 200
    assert response.json()["session_id"] == "app-session-id"
    assert response.json()["user_id"] == "8bd04cb8-8ee8-40b6-bdb3-22b738920730"
    created_user = manager.create_session.await_args.kwargs["user"]
    assert created_user.display_name == "Jeevan"
    assert session.metadata["auth_provider"] == "supabase"
    manager.update_session.assert_awaited_once_with(session)


def test_invalid_supabase_access_token_is_rejected(monkeypatch) -> None:
    settings = SimpleNamespace(
        auth_enabled=True,
        auth_provider="supabase",
        supabase_url="https://example.supabase.co",
        supabase_anon_key=SecretStr("anon-key"),
        session_expiry_hours=24,
    )
    monkeypatch.setattr(routes, "get_settings", lambda: settings)

    def create_http_client(*, timeout: float) -> InvalidSupabaseAuthHttpClient:
        assert timeout > 0
        return InvalidSupabaseAuthHttpClient()

    monkeypatch.setattr(routes.httpx, "AsyncClient", create_http_client)

    app = FastAPI()
    app.state.limiter = routes.limiter
    app.include_router(routes.router)

    with TestClient(app) as client:
        response = client.post(
            "/auth/supabase/session",
            headers={"Authorization": "Bearer expired-token"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "The Supabase access token is invalid or expired."
