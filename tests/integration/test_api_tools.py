"""Integration tests for /api/v1/tools and /api/v1/tools/execute."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.config.settings import Settings, clear_settings_cache
from src.mcp_tools.base import ToolResult, ToolStatus
from src.middleware.rate_limit import reset_rate_limiter
from src.models.user import SessionContext, TokenInfo, UserProfile
from src.session.manager import SessionManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
async def tools_app_client(monkeypatch):
    """Create an ASGI test client with AUTH_ENABLED=false for tool tests.

    With authentication disabled the ``get_current_session`` dependency
    returns a mock ``SessionContext`` with
    ``selected_org="https://dev.azure.com/TestOrg"`` which passes the
    SSRF validation in the ``execute_tool`` route.

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
    monkeypatch.setenv("AZURE_DEVOPS_DEFAULT_ORG_URL", "https://dev.azure.com/TestOrg")
    monkeypatch.setenv(
        "ENCRYPTION_KEY",
        "dGVzdC1lbmNyeXB0aW9uLWtleS0xMjM0NTY3ODkwYWI=",
    )

    clear_settings_cache()
    reset_rate_limiter()

    from src.main import create_app

    app = create_app()

    # Manually initialise app.state (lifespan does not run with ASGITransport)
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


@pytest.fixture()
async def auth_tools_app_client(monkeypatch):
    """Create an ASGI test client with AUTH_ENABLED=true for auth-required tool tests.

    Used to verify that tool execution requires an authenticated session
    when authentication is enabled.

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

        # Manually initialise app.state
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


def _make_mock_session(
    selected_org: str = "https://dev.azure.com/TestOrg",
) -> SessionContext:
    """Build a mock ``SessionContext`` for dependency overrides.

    Args:
        selected_org: The organization URL to set on the session.

    Returns:
        A ``SessionContext`` with predictable test data.
    """
    return SessionContext(
        session_id="sess-integration-test",
        user=UserProfile(
            id="user-integration-001",
            display_name="Integration Test User",
            email="integration@test.com",
            user_principal_name="integration@test.com",
            tenant_id="00000000-0000-0000-0000-000000000000",
        ),
        token_info=TokenInfo(
            access_token="test-access-token",
            refresh_token="test-refresh-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            scope="openid profile",
        ),
        selected_org=selected_org,
        selected_project="TestProject",
    )


# =============================================================================
# Tests -- GET /api/v1/tools
# =============================================================================


async def test_list_tools_returns_tools(tools_app_client) -> None:
    """GET /api/v1/tools returns a non-empty list of tool definitions."""
    client, _app = tools_app_client

    response = await client.get("/api/v1/tools")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) > 0


async def test_list_tools_each_has_required_fields(tools_app_client) -> None:
    """Every tool in GET /api/v1/tools has name, description, category, and parameters."""
    client, _app = tools_app_client

    response = await client.get("/api/v1/tools")
    assert response.status_code == 200

    for tool in response.json():
        assert "name" in tool, f"Tool missing 'name': {tool}"
        assert "description" in tool, f"Tool missing 'description': {tool}"
        assert "category" in tool, f"Tool missing 'category': {tool}"
        assert "parameters" in tool, f"Tool missing 'parameters': {tool}"
        assert isinstance(tool["parameters"], list)


# =============================================================================
# Tests -- POST /api/v1/tools/execute
# =============================================================================


async def test_execute_tool_not_found(tools_app_client) -> None:
    """POST /api/v1/tools/execute with an unknown tool_name returns 404."""
    client, _app = tools_app_client

    response = await client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "nonexistent_tool_xyz", "parameters": {}},
    )

    assert response.status_code == 404
    assert "Tool not found" in response.json()["detail"]


async def test_execute_tool_requires_session(auth_tools_app_client) -> None:
    """When AUTH_ENABLED=true and no session header is provided, execute returns 401."""
    client, _app = auth_tools_app_client

    response = await client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "get_work_item", "parameters": {}},
    )

    assert response.status_code == 401


async def test_execute_tool_ssrf_prevention(tools_app_client) -> None:
    """A session with a non-Azure org URL triggers the SSRF check and returns 400.

    The ``get_current_session`` dependency is overridden to inject a
    session whose ``selected_org`` points to an untrusted host.
    """
    client, app = tools_app_client

    evil_session = _make_mock_session(selected_org="http://evil.com")

    # Override the dependency to return a session with an untrusted org URL
    from src.auth.dependencies import get_current_session

    async def _override_session() -> SessionContext:
        return evil_session

    app.dependency_overrides[get_current_session] = _override_session

    try:
        # Use a real tool name from the list so we get past the 404 check
        tools_response = await client.get("/api/v1/tools")
        first_tool_name = tools_response.json()[0]["name"]

        response = await client.post(
            "/api/v1/tools/execute",
            json={"tool_name": first_tool_name, "parameters": {}},
        )

        assert response.status_code == 400
        assert "Invalid organization URL" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_current_session, None)


async def test_execute_tool_success_mock(tools_app_client) -> None:
    """Mock a tool's execute method to verify the response format.

    Patches ``create_all_tools`` to return a single tool whose execute
    method returns a known ``ToolResult``.  Verifies that the response
    contains ``status``, ``data``, and ``message`` fields.
    """
    client, _app = tools_app_client

    mock_tool = MagicMock()
    mock_tool.definition.name = "mock_test_tool"
    mock_tool.definition.description = "A mock tool for testing"
    mock_tool.definition.category = "test"
    mock_tool.definition.parameters = []
    mock_tool.execute = AsyncMock(
        return_value=ToolResult.success(
            data={"result_key": "result_value"},
            message="Mock tool executed successfully",
        )
    )

    with patch("src.api.routes.create_all_tools", return_value=[mock_tool]):
        response = await client.post(
            "/api/v1/tools/execute",
            json={"tool_name": "mock_test_tool", "parameters": {}},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"] == {"result_key": "result_value"}
    assert body["message"] == "Mock tool executed successfully"


async def test_execute_tool_context_param_renamed(tools_app_client) -> None:
    """When parameters include 'context', it is renamed to 'additional_context'.

    The ``execute_tool`` route renames the ``context`` parameter to
    ``additional_context`` to avoid a keyword conflict with the
    ``ToolContext`` argument in ``BaseTool.execute()``.
    """
    client, _app = tools_app_client

    mock_tool = MagicMock()
    mock_tool.definition.name = "mock_context_tool"
    mock_tool.definition.description = "Tool that accepts context param"
    mock_tool.definition.category = "test"
    mock_tool.definition.parameters = []
    mock_tool.execute = AsyncMock(
        return_value=ToolResult.success(data={"ok": True})
    )

    with patch("src.api.routes.create_all_tools", return_value=[mock_tool]):
        response = await client.post(
            "/api/v1/tools/execute",
            json={
                "tool_name": "mock_context_tool",
                "parameters": {"context": "some extra context text"},
            },
        )

    assert response.status_code == 200

    # Verify the tool was called with 'additional_context' instead of 'context'
    _call_args, call_kwargs = mock_tool.execute.call_args
    assert "additional_context" in call_kwargs
    assert call_kwargs["additional_context"] == "some extra context text"
    assert "context" not in call_kwargs or not isinstance(call_kwargs.get("context"), str)


async def test_execute_tool_value_error_returns_400(tools_app_client) -> None:
    """When a tool raises ``ValueError``, the route returns HTTP 400."""
    client, _app = tools_app_client

    mock_tool = MagicMock()
    mock_tool.definition.name = "mock_validation_tool"
    mock_tool.definition.description = "Tool that raises ValueError"
    mock_tool.definition.category = "test"
    mock_tool.definition.parameters = []
    mock_tool.execute = AsyncMock(side_effect=ValueError("work_item_id is required"))

    with patch("src.api.routes.create_all_tools", return_value=[mock_tool]):
        response = await client.post(
            "/api/v1/tools/execute",
            json={"tool_name": "mock_validation_tool", "parameters": {}},
        )

    assert response.status_code == 400
    assert "work_item_id is required" in response.json()["detail"]


async def test_execute_tool_logs_traceback_for_unexpected_error(tools_app_client, caplog) -> None:
    """Unexpected tool errors are logged with traceback but hidden from clients."""
    client, _app = tools_app_client

    mock_tool = MagicMock()
    mock_tool.definition.name = "mock_unexpected_error_tool"
    mock_tool.definition.description = "Tool that raises an unexpected exception"
    mock_tool.definition.category = "test"
    mock_tool.definition.parameters = []
    mock_tool.execute = AsyncMock(side_effect=RuntimeError())

    with patch("src.api.routes.create_all_tools", return_value=[mock_tool]):
        response = await client.post(
            "/api/v1/tools/execute",
            json={"tool_name": "mock_unexpected_error_tool", "parameters": {}},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Tool execution failed. Please try again."
    assert "Tool execution failed" in caplog.text
    assert "RuntimeError" in caplog.text
