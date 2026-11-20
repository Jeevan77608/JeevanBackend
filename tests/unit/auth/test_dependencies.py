"""Tests for src/auth/dependencies.py — auth dependencies and session resolution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import HTTPException

from src.auth.dependencies import (
    _validate_internal_key,
    create_mock_session,
    extract_fingerprint,
    get_client_ip,
    get_current_session,
    get_session_manager,
)
from src.models.user import SessionContext
from src.session.exceptions import SessionNotFoundError
from src.session.models import SessionFingerprint


# =============================================================================
# Helpers
# =============================================================================


class _HeadersDict(dict):
    """Dict subclass whose .get is assignable (for mocking FastAPI Headers)."""
    pass


def _make_request(
    headers: dict[str, str] | None = None,
    client_host: str | None = None,
    app_state: dict | None = None,
) -> MagicMock:
    """Build a mock FastAPI Request."""
    request = MagicMock()

    _headers = _HeadersDict(headers or {})
    request.headers = _headers

    if client_host is not None:
        request.client = MagicMock()
        request.client.host = client_host
    else:
        request.client = None

    if app_state is not None:
        state = MagicMock()
        for k, v in app_state.items():
            setattr(state, k, v)
        request.app.state = state
    else:
        # spec=[] means no attributes exist
        request.app.state = MagicMock(spec=[])

    return request


# =============================================================================
# extract_fingerprint
# =============================================================================


class TestExtractFingerprint:
    """Tests for browser-fingerprint extraction."""

    def test_extract_fingerprint_with_user_agent(self) -> None:
        fp = extract_fingerprint(user_agent="Mozilla/5.0 Test")
        assert fp is not None
        assert isinstance(fp, SessionFingerprint)

    def test_extract_fingerprint_none_when_no_user_agent(self) -> None:
        fp = extract_fingerprint(user_agent=None)
        assert fp is None

    def test_extract_fingerprint_with_all_fields(self) -> None:
        fp = extract_fingerprint(
            user_agent="Mozilla/5.0 Test",
            client_ip="192.168.1.1",
            accept_language="en-US,en;q=0.9",
        )
        assert fp is not None
        assert isinstance(fp, SessionFingerprint)


# =============================================================================
# get_client_ip
# =============================================================================


class TestGetClientIP:
    """Tests for client IP extraction."""

    def test_get_client_ip_from_forwarded_for(self) -> None:
        request = _make_request(
            headers={"X-Forwarded-For": "10.0.0.1, 10.0.0.2"},
            client_host="127.0.0.1",
        )
        ip = get_client_ip(request)
        assert ip == "10.0.0.1"

    def test_get_client_ip_from_client_host(self) -> None:
        request = _make_request(client_host="192.168.0.10")
        ip = get_client_ip(request)
        assert ip == "192.168.0.10"

    def test_get_client_ip_none_when_no_client(self) -> None:
        request = _make_request()
        ip = get_client_ip(request)
        assert ip is None


# =============================================================================
# create_mock_session
# =============================================================================


class TestCreateMockSession:
    """Tests for the dev-mode mock session factory."""

    @patch("src.auth.dependencies.get_settings")
    def test_create_mock_session_returns_session_context(
        self, mock_get_settings: MagicMock
    ) -> None:
        settings = MagicMock()
        settings.azure_devops.pat = None
        settings.azure_devops.default_org_url = "https://dev.azure.com/TestOrg"
        settings.azure_devops.default_project = "TestProject"
        mock_get_settings.return_value = settings

        session = create_mock_session()
        assert isinstance(session, SessionContext)
        assert session.session_id == "dev-session-no-auth"

    @patch("src.auth.dependencies.get_settings")
    def test_create_mock_session_uses_pat_when_available(
        self, mock_get_settings: MagicMock
    ) -> None:
        settings = MagicMock()
        settings.azure_devops.pat.get_secret_value.return_value = "my-pat-token"
        settings.azure_devops.default_org_url = "https://dev.azure.com/Org"
        settings.azure_devops.default_project = "Proj"
        mock_get_settings.return_value = settings

        session = create_mock_session()
        assert session.token_info.access_token == "my-pat-token"


# =============================================================================
# _validate_internal_key
# =============================================================================


class TestValidateInternalKey:
    """Tests for server-to-server internal API key validation."""

    @patch("src.auth.dependencies.get_settings")
    def test_validate_internal_key_success(self, mock_get_settings: MagicMock) -> None:
        settings = MagicMock()
        settings.internal_api_key.get_secret_value.return_value = "super-secret-key"
        mock_get_settings.return_value = settings

        request = _make_request(headers={"X-Internal-Key": "super-secret-key"})
        assert _validate_internal_key(request) is True

    @patch("src.auth.dependencies.get_settings")
    def test_validate_internal_key_failure_wrong_key(
        self, mock_get_settings: MagicMock
    ) -> None:
        settings = MagicMock()
        settings.internal_api_key.get_secret_value.return_value = "super-secret-key"
        mock_get_settings.return_value = settings

        request = _make_request(headers={"X-Internal-Key": "wrong-key"})
        assert _validate_internal_key(request) is False

    @patch("src.auth.dependencies.get_settings")
    def test_validate_internal_key_no_key_configured_allows(
        self, mock_get_settings: MagicMock
    ) -> None:
        settings = MagicMock()
        settings.internal_api_key.get_secret_value.return_value = ""
        mock_get_settings.return_value = settings

        request = _make_request()
        assert _validate_internal_key(request) is True


# =============================================================================
# get_session_manager
# =============================================================================


class TestGetSessionManager:
    """Tests for the session manager dependency."""

    async def test_get_session_manager_returns_from_app_state(self) -> None:
        mock_manager = MagicMock()
        request = _make_request(app_state={"session_manager": mock_manager})
        result = await get_session_manager(request)
        assert result is mock_manager

    async def test_get_session_manager_raises_503_when_missing(self) -> None:
        request = _make_request()
        with pytest.raises(HTTPException) as exc_info:
            await get_session_manager(request)
        assert exc_info.value.status_code == 503


# =============================================================================
# get_current_session
# =============================================================================


class TestGetCurrentSession:
    """Tests for the mandatory session dependency."""

    @patch("src.auth.dependencies.get_settings")
    async def test_get_current_session_auth_disabled_returns_mock(
        self, mock_get_settings: MagicMock
    ) -> None:
        settings = MagicMock()
        settings.auth_enabled = False
        settings.azure_devops.pat = None
        settings.azure_devops.default_org_url = "https://dev.azure.com/TestOrg"
        settings.azure_devops.default_project = "TestProject"
        mock_get_settings.return_value = settings

        request = _make_request()
        mock_manager = AsyncMock()

        session = await get_current_session(
            request=request,
            session_id=None,
            x_session_id=None,
            user_agent=None,
            accept_language=None,
            session_manager=mock_manager,
        )
        assert isinstance(session, SessionContext)

    @patch("src.auth.dependencies.get_settings")
    async def test_get_current_session_no_session_id_raises_401(
        self, mock_get_settings: MagicMock
    ) -> None:
        settings = MagicMock()
        settings.auth_enabled = True
        mock_get_settings.return_value = settings

        request = _make_request()
        mock_manager = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_current_session(
                request=request,
                session_id=None,
                x_session_id=None,
                user_agent=None,
                accept_language=None,
                session_manager=mock_manager,
            )
        assert exc_info.value.status_code == 401

    @patch("src.auth.dependencies.get_settings")
    async def test_get_current_session_session_not_found_raises_401(
        self, mock_get_settings: MagicMock
    ) -> None:
        settings = MagicMock()
        settings.auth_enabled = True
        settings.internal_api_key.get_secret_value.return_value = ""
        mock_get_settings.return_value = settings

        request = _make_request(headers={"User-Agent": "TestAgent"})
        mock_manager = AsyncMock()
        mock_manager.get_session.side_effect = SessionNotFoundError("not found")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_session(
                request=request,
                session_id=None,
                x_session_id="nonexistent-session",
                user_agent="TestAgent",
                accept_language=None,
                session_manager=mock_manager,
            )
        assert exc_info.value.status_code == 401
