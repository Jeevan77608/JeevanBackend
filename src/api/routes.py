"""
REST API routes for the Azure DevOps MCP Server.

This module defines all REST API endpoints organized by functionality:
- Health: Server health and readiness checks
- Auth: OAuth 2.0 authentication flow with Azure AD
- Session: User session management
- Tools: MCP tool discovery and execution
- Drafts: Draft work item management

Each endpoint is documented with OpenAPI metadata for Swagger UI and
protected by rate limiting where appropriate.

Example:
    The routes are mounted at /api/v1 prefix in main.py:

    >>> # Health check
    >>> GET /api/v1/health

    >>> # List available tools
    >>> GET /api/v1/tools

    >>> # Execute a tool
    >>> POST /api/v1/tools/execute
    >>> {"tool_name": "get_work_item", "parameters": {"work_item_id": 123}}
"""

from __future__ import annotations

import importlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]

from src.auth.dependencies import get_current_session, get_oauth_provider
from src.auth.oauth import AzureADOAuthProvider
from src.auth.google_oauth import GoogleOAuthError, GoogleOAuthProvider
from src.config.settings import get_settings
from src.integrations.jira import JiraCloudClient, JiraIntegrationError
from src.mcp_tools import ToolContext, create_all_tools
from src.middleware.rate_limit import (
    DEFAULT_RATE_LIMITS,
    get_rate_limiter,
    rate_limit_key_func,
    rate_limit_key_ip_only,
)
from src.models.user import SessionContext, UserProfile
from src.services.file_processor import FileProcessor
from src.session.manager import SessionManager

logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Rate limiter (singleton — same instance is stored in app.state by main.py)
limiter = get_rate_limiter()

# Allowed MIME types for file upload
ALLOWED_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}


# =============================================================================
# Request/Response Models (Pydantic)
# =============================================================================


class AuthStartRequest(BaseModel):
    """Request to initiate OAuth 2.0 authentication flow."""

    redirect_uri: str = Field(
        ...,
        description="URI to redirect after authentication completes",
        examples=["https://your-frontend.azurewebsites.net/api/auth/callback"],
    )

    model_config = {"json_schema_extra": {"example": {"redirect_uri": "https://your-frontend.azurewebsites.net/api/auth/callback"}}}


class AuthStartResponse(BaseModel):
    """Response containing OAuth authorization URL."""

    authorization_url: str = Field(..., description="URL to redirect user for authentication")
    state: str = Field(..., description="CSRF protection state parameter")


class AuthCallbackRequest(BaseModel):
    """OAuth callback request with authorization code."""

    code: str = Field(..., description="Authorization code from OAuth provider")
    state: str = Field(..., description="State parameter for CSRF validation")


class AuthCallbackResponse(BaseModel):
    """Response after successful OAuth callback."""

    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="Azure AD user ID")
    expires_at: str = Field(..., description="Session expiration timestamp (ISO 8601)")


class GoogleAuthCallbackRequest(BaseModel):
    """Google OAuth callback values."""

    code: str
    state: str


class JiraIssueCreateRequest(BaseModel):
    """Fields required to create a Jira issue."""

    project_key: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    summary: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=10000)
    issue_type: str = Field(default="Task", min_length=1, max_length=64)


class JiraCommentRequest(BaseModel):
    """Text for a Jira issue comment."""

    text: str = Field(..., min_length=1, max_length=10000)


class ToolExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""

    tool_name: str = Field(
        ...,
        description="Name of the tool to execute",
        examples=["get_work_item", "generate_user_stories"],
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific parameters",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Current saved conversation id; when set, draft changes are persisted to this conversation immediately",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tool_name": "get_work_item",
                "parameters": {"work_item_id": 1234},
            }
        }
    }


class ToolExecuteResponse(BaseModel):
    """Response from tool execution."""

    status: str = Field(..., description="Execution status: success, error, or partial")
    data: Any = Field(default=None, description="Tool execution result data")
    message: str | None = Field(default=None, description="Human-readable message")
    error: str | None = Field(default=None, description="Error message if status is error")


class HealthResponse(BaseModel):
    """Health check response for monitoring."""

    status: str = Field(..., description="Server status: healthy, degraded, or unhealthy")
    version: str = Field(..., description="API version")
    redis_connected: bool = Field(..., description="Redis connection status")
    cosmos_connected: bool = Field(
        ...,
        description="Cosmos DB connection status (true when CHAT_SAVE_ENABLED and Cosmos reachable, else false)",
    )


class ConfigResponse(BaseModel):
    """Public config for frontend (feature flags, chat scope, context limits)."""

    chat_save_enabled: bool = Field(
        ..., description="Whether conversation history (save/list/load) is enabled"
    )
    chat_scope_strict: bool = Field(
        default=True,
        description="Whether to restrict chat to epic/task generation only",
    )
    chat_scope_boundary_message: str = Field(
        default="",
        description="Message shown when user message is out of scope",
    )
    chat_context_window_messages: int = Field(
        default=20,
        ge=2,
        le=100,
        description="Max messages (including system) sent to the LLM",
    )
    chat_history_display_days: int = Field(
        default=15,
        ge=1,
        le=365,
        description="Number of days of conversation history to show in sidebar",
    )
    chat_history_limit_messages: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max history messages (user + assistant) to include per request",
    )
    default_push_tag: str = Field(
        default="@AIGenerated",
        description="Default tag injected on every push to Azure DevOps",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "redis_connected": True,
            }
        }
    }


# =============================================================================
# Health Endpoints
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health Check",
    description="Check server health status including Redis connectivity.",
    responses={
        200: {"description": "Server health status"},
    },
)
async def health_check(request: Request) -> HealthResponse:
    """
    Check server health and readiness.

    Returns the current health status of the server including:
    - Overall server status
    - API version
    - Redis/session store connectivity

    This endpoint is suitable for Kubernetes liveness/readiness probes.
    """
    session_manager: SessionManager = request.app.state.session_manager
    settings = get_settings()

    redis_ok = False
    try:
        health_result = await session_manager.health_check()
        redis_ok = health_result.get("status") == "healthy"
    except Exception:
        pass

    cosmos_ok = False
    if settings.chat_save_enabled:
        try:
            from src.clients.cosmos import get_conversations_database

            db = get_conversations_database()
            if db is not None:
                db.read()  # verify Cosmos connectivity
                cosmos_ok = True
        except Exception:
            pass  # cosmos_ok stays False

    return HealthResponse(
        status="healthy" if redis_ok else "degraded",
        version="1.0.0",
        redis_connected=redis_ok,
        cosmos_connected=cosmos_ok,
    )


@router.get(
    "/config",
    response_model=ConfigResponse,
    tags=["health"],
    summary="Frontend config",
    description="Public feature flags (e.g. chat_save_enabled) for UI.",
)
async def get_config() -> ConfigResponse:
    """Return config the frontend needs (feature flags, chat scope, context limits)."""
    settings = get_settings()
    return ConfigResponse(
        chat_save_enabled=settings.chat_save_enabled,
        chat_scope_strict=settings.chat_scope_strict,
        chat_scope_boundary_message=settings.chat_scope_boundary_message,
        chat_context_window_messages=settings.chat_context_window_messages,
        chat_history_display_days=settings.chat_history_display_days,
        chat_history_limit_messages=settings.chat_history_limit_messages,
        default_push_tag=settings.default_push_tag,
    )


# =============================================================================
# Auth Endpoints
# =============================================================================


@router.post("/auth/supabase/session", tags=["auth"])
@limiter.limit(DEFAULT_RATE_LIMITS.auth, key_func=rate_limit_key_ip_only)
async def create_supabase_session(request: Request) -> dict[str, str]:
    """Validate a Supabase access token and create the app's encrypted session."""
    settings = get_settings()
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Account sign-in is not enabled.",
        )
    if (
        not settings.supabase_url
        or not settings.supabase_anon_key.get_secret_value()
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email/password sign-in needs Supabase URL and anonymous key configured.",
        )

    scheme, _, access_token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A Supabase access token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
                headers={
                    "apikey": settings.supabase_anon_key.get_secret_value(),
                    "Authorization": f"Bearer {access_token}",
                },
            )
    except httpx.HTTPError as error:
        logger.exception("Could not verify Supabase access token")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify sign-in with Supabase.",
        ) from error

    if response.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase sign-in verification is temporarily unavailable.",
        )
    if not response.is_success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The Supabase access token is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        profile = response.json()
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Supabase returned an invalid user profile.",
        ) from error

    user_metadata = profile.get("user_metadata") or {}
    email = profile.get("email") or ""
    user = UserProfile(
        id=str(profile["id"]),
        display_name=(
            user_metadata.get("full_name")
            or user_metadata.get("name")
            or user_metadata.get("user_name")
            or email
            or "Jeevan AI Chatbot user"
        ),
        email=email,
        user_principal_name=email,
        tenant_id="supabase",
    )

    session_manager: SessionManager = request.app.state.session_manager
    session = await session_manager.create_session(user=user)
    session.metadata["auth_provider"] = "supabase"
    await session_manager.update_session(session)

    expires_at = datetime.now(timezone.utc) + timedelta(
        hours=settings.session_expiry_hours
    )
    return {
        "session_id": session.session_id,
        "user_id": user.id,
        "expires_at": expires_at.isoformat(),
    }


@router.post("/auth/google/start", response_model=AuthStartResponse, tags=["auth"])
@limiter.limit(DEFAULT_RATE_LIMITS.auth, key_func=rate_limit_key_ip_only)
async def google_auth_start(request: Request) -> AuthStartResponse:
    """Start Google OpenID Connect sign-in."""
    provider: GoogleOAuthProvider | None = getattr(
        request.app.state, "google_oauth_provider", None
    )
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google sign-in is not configured.",
        )

    auth_state = provider.create_authorization_state()
    state_data = auth_state.to_dict()
    state_data["provider"] = "google"
    await request.app.state.session_manager.save_auth_state(
        state=auth_state.state,
        data=state_data,
        ttl_seconds=600,
    )
    return AuthStartResponse(
        authorization_url=provider.get_authorization_url(auth_state),
        state=auth_state.state,
    )


@router.post("/auth/google/callback", response_model=AuthCallbackResponse, tags=["auth"])
@limiter.limit(DEFAULT_RATE_LIMITS.auth, key_func=rate_limit_key_ip_only)
async def google_auth_callback(
    request: Request,
    request_body: GoogleAuthCallbackRequest,
) -> AuthCallbackResponse:
    """Validate Google identity and create an application session."""
    provider: GoogleOAuthProvider | None = getattr(
        request.app.state, "google_oauth_provider", None
    )
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google sign-in is not configured.",
        )

    session_manager: SessionManager = request.app.state.session_manager
    auth_state_data = await session_manager.get_auth_state(request_body.state)
    if not auth_state_data or auth_state_data.get("provider") != "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired Google sign-in state. Please try again.",
        )

    try:
        from src.auth.oauth import AuthorizationState

        auth_state = AuthorizationState.from_dict(auth_state_data)
        user, token_info = await provider.exchange_code_for_user(
            request_body.code, auth_state
        )
        session = await session_manager.create_session(user, token_info)
        session.metadata["auth_provider"] = "google"
        await session_manager.update_session(session)
        return AuthCallbackResponse(
            session_id=session.session_id,
            user_id=session.user.id,
            expires_at=(
                session.created_at
                + timedelta(hours=get_settings().session_expiry_hours)
            ).isoformat(),
        )
    except GoogleOAuthError as error:
        logger.warning("Google sign-in failed: %s", error)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.error("Google sign-in callback failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google sign-in could not be completed. Please try again.",
        ) from error


def _jira_client() -> JiraCloudClient:
    return JiraCloudClient(get_settings())


def _require_jira_auth(session: SessionContext) -> None:
    settings = get_settings()
    if not settings.auth_enabled or settings.auth_provider not in {"google", "supabase"}:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Jira requires an enabled account sign-in provider.",
        )
    if session.metadata.get("auth_provider") not in {"google", "supabase"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in before connecting Jira.",
        )


def _raise_jira_error(error: JiraIntegrationError) -> None:
    message = str(error)
    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if "not configured" in message
        else status.HTTP_400_BAD_REQUEST
    )
    raise HTTPException(status_code=status_code, detail=message) from error


@router.get("/jira/status", tags=["jira"])
async def jira_connection_status(
    session: SessionContext = Depends(get_current_session),
) -> dict[str, Any]:
    """Return whether the signed-in user connected their Jira account."""
    _require_jira_auth(session)
    connection = session.integration_tokens.get("jira")
    if not connection:
        return {"connected": False}
    return {
        "connected": True,
        "site_name": connection.get("site_name"),
        "site_url": connection.get("site_url"),
        "account_name": connection.get("account_name"),
    }


@router.post("/jira/connect/start", tags=["jira"])
@limiter.limit(DEFAULT_RATE_LIMITS.auth, key_func=rate_limit_key_ip_only)
async def jira_connect_start(
    request: Request,
    session: SessionContext = Depends(get_current_session),
) -> dict[str, str]:
    """Start a per-user Atlassian OAuth 2.0 authorization flow."""
    _require_jira_auth(session)
    client = _jira_client()
    state = client.new_state()
    try:
        authorization_url = client.create_authorization_url(state)
    except JiraIntegrationError as error:
        _raise_jira_error(error)
    await request.app.state.session_manager.save_auth_state(
        state,
        {"provider": "jira", "session_id": session.session_id},
        ttl_seconds=600,
    )
    return {"authorization_url": authorization_url}


@router.get("/jira/connect/callback", tags=["jira"])
@limiter.limit(DEFAULT_RATE_LIMITS.auth, key_func=rate_limit_key_ip_only)
async def jira_connect_callback(
    request: Request,
    code: str,
    state: str,
    session: SessionContext = Depends(get_current_session),
) -> dict[str, Any]:
    """Complete per-user Atlassian authorization and store encrypted tokens."""
    _require_jira_auth(session)
    session_manager: SessionManager = request.app.state.session_manager
    state_data = await session_manager.get_auth_state(state)
    if (
        not state_data
        or state_data.get("provider") != "jira"
        or state_data.get("session_id") != session.session_id
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired Jira connection state. Please try again.",
        )
    try:
        connection = await _jira_client().exchange_code(code)
        session.integration_tokens["jira"] = connection
        await session_manager.update_session(session)
        try:
            jira_user = await _jira_client().request(
                session, session_manager, "GET", "myself"
            )
            connection["account_name"] = jira_user.get("displayName")
            session.integration_tokens["jira"] = connection
            await session_manager.update_session(session)
        except JiraIntegrationError:
            logger.info("Jira connected; profile details are unavailable for this account")
        return {
            "connected": True,
            "site_name": connection.get("site_name"),
            "site_url": connection.get("site_url"),
            "account_name": connection.get("account_name"),
        }
    except JiraIntegrationError as error:
        _raise_jira_error(error)


@router.delete("/jira/connect", tags=["jira"])
@limiter.limit(DEFAULT_RATE_LIMITS.tool_execution, key_func=rate_limit_key_func)
async def jira_disconnect(
    request: Request,
    session: SessionContext = Depends(get_current_session),
) -> dict[str, bool]:
    """Disconnect the current user's Jira account."""
    _require_jira_auth(session)
    session.integration_tokens.pop("jira", None)
    await request.app.state.session_manager.update_session(session)
    return {"connected": False}


@router.get("/jira/projects", tags=["jira"])
@limiter.limit(DEFAULT_RATE_LIMITS.tool_execution, key_func=rate_limit_key_func)
async def jira_projects(
    request: Request,
    session: SessionContext = Depends(get_current_session),
) -> dict[str, Any]:
    """List Jira projects visible to the connected user."""
    _require_jira_auth(session)
    try:
        return await _jira_client().request(
            session,
            request.app.state.session_manager,
            "GET",
            "project/search",
            params={"maxResults": 100, "orderBy": "name"},
        )
    except JiraIntegrationError as error:
        _raise_jira_error(error)


@router.get("/jira/issues", tags=["jira"])
@limiter.limit(DEFAULT_RATE_LIMITS.tool_execution, key_func=rate_limit_key_func)
async def jira_search_issues(
    request: Request,
    jql: str | None = Query(default=None, max_length=2000),
    session: SessionContext = Depends(get_current_session),
) -> dict[str, Any]:
    """Search Jira issues using Jira Query Language."""
    _require_jira_auth(session)
    effective_jql = jql or (
        f"project = {session.selected_project} ORDER BY updated DESC"
        if session.selected_project
        else None
    )
    if not effective_jql:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Select a Jira project or provide a project-scoped JQL query.",
        )
    try:
        return await _jira_client().request(
            session,
            request.app.state.session_manager,
            "GET",
            "search/jql",
            params={
                "jql": effective_jql,
                "maxResults": 50,
                "fields": "key,summary,status,issuetype,project,updated",
            },
        )
    except JiraIntegrationError as error:
        _raise_jira_error(error)


@router.post("/jira/issues", tags=["jira"])
@limiter.limit(DEFAULT_RATE_LIMITS.tool_execution, key_func=rate_limit_key_func)
async def jira_create_issue(
    request: Request,
    issue: JiraIssueCreateRequest,
    session: SessionContext = Depends(get_current_session),
) -> dict[str, Any]:
    """Create an issue in a project the connected Jira user can access."""
    _require_jira_auth(session)
    description = (
        {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": issue.description}],
                }
            ],
        }
        if issue.description
        else None
    )
    fields: dict[str, Any] = {
        "project": {"key": issue.project_key},
        "summary": issue.summary,
        "issuetype": {"name": issue.issue_type},
    }
    if description:
        fields["description"] = description
    try:
        return await _jira_client().request(
            session,
            request.app.state.session_manager,
            "POST",
            "issue",
            json_body={"fields": fields},
        )
    except JiraIntegrationError as error:
        _raise_jira_error(error)


@router.post("/jira/issues/{issue_key}/comments", tags=["jira"])
@limiter.limit(DEFAULT_RATE_LIMITS.tool_execution, key_func=rate_limit_key_func)
async def jira_add_comment(
    issue_key: str,
    comment: JiraCommentRequest,
    request: Request,
    session: SessionContext = Depends(get_current_session),
) -> dict[str, Any]:
    """Add a comment to a Jira issue the connected user can access."""
    _require_jira_auth(session)
    if not re.fullmatch(r"[A-Za-z0-9_-]+-\d+", issue_key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Jira issue key.",
        )
    body = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": comment.text}],
                }
            ],
        }
    }
    try:
        return await _jira_client().request(
            session,
            request.app.state.session_manager,
            "POST",
            f"issue/{issue_key}/comment",
            json_body=body,
        )
    except JiraIntegrationError as error:
        _raise_jira_error(error)


@router.post("/auth/start", response_model=AuthStartResponse, tags=["auth"])
@limiter.limit(DEFAULT_RATE_LIMITS.auth, key_func=rate_limit_key_ip_only)
async def auth_start(
    request: Request,
    request_body: AuthStartRequest,
    oauth_provider: AzureADOAuthProvider = Depends(get_oauth_provider),
) -> AuthStartResponse:
    """Start OAuth authentication flow with PKCE."""
    if not oauth_provider:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth provider not available. Authentication may be disabled.",
        )

    session_manager: SessionManager = request.app.state.session_manager

    # Create authorization state with PKCE (use redirect_uri from request if provided)
    redirect_uri = request_body.redirect_uri or None

    # Validate redirect_uri against configured allowed URI
    if redirect_uri:
        settings = get_settings()
        configured_redirect = settings.azure_ad.redirect_uri if settings.azure_ad else None
        if configured_redirect and redirect_uri != configured_redirect:
            logger.warning(f"Rejected redirect_uri: {redirect_uri} (expected: {configured_redirect})")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid redirect URI",
            )

    auth_state = oauth_provider.create_authorization_state(redirect_uri=redirect_uri)

    # Log PKCE details for debugging
    logger.info(
        f"Created auth state. State: {auth_state.state[:16]}..., "
        f"Verifier length: {len(auth_state.pkce.code_verifier)}, "
        f"Challenge: {auth_state.pkce.code_challenge[:16]}..., "
        f"Redirect URI: {auth_state.redirect_uri}"
    )

    # Store state temporarily (10 minutes TTL)
    state_dict = auth_state.to_dict()
    await session_manager.save_auth_state(
        state=auth_state.state,
        data=state_dict,
        ttl_seconds=600,
    )

    # Verify what we stored
    logger.debug(f"Stored auth state data keys: {list(state_dict.keys())}")

    # Generate authorization URL
    auth_url = oauth_provider.get_authorization_url(auth_state)

    # Log authorization URL for debugging
    logger.info(f"Generated authorization URL. Redirect URI: {auth_state.redirect_uri}")

    return AuthStartResponse(
        authorization_url=auth_url,
        state=auth_state.state,
    )


@router.post("/auth/callback", response_model=AuthCallbackResponse, tags=["auth"])
@limiter.limit(DEFAULT_RATE_LIMITS.auth, key_func=rate_limit_key_ip_only)
async def auth_callback(
    request: Request,
    request_body: AuthCallbackRequest,
    oauth_provider: AzureADOAuthProvider = Depends(get_oauth_provider),
) -> AuthCallbackResponse:
    """Handle OAuth callback and create session."""
    if not oauth_provider:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth provider not available. Authentication may be disabled.",
        )

    session_manager: SessionManager = request.app.state.session_manager

    try:
        # Retrieve and validate state (one-time use, prevents CSRF)
        logger.debug(f"Retrieving auth state for state: {request_body.state[:16]}...")
        auth_state_data = await session_manager.get_auth_state(request_body.state)
        if not auth_state_data:
            logger.warning(f"Auth state not found or expired for state: {request_body.state[:16]}...")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state parameter. Please try logging in again.",
            )

        # Reconstruct authorization state from stored data
        from src.auth.oauth import AuthorizationState
        auth_state = AuthorizationState.from_dict(auth_state_data)
        logger.debug(
            f"Auth state retrieved successfully. "
            f"Code verifier length: {len(auth_state.pkce.code_verifier)}, "
            f"Code challenge length: {len(auth_state.pkce.code_challenge)}"
        )

        # Exchange authorization code for tokens (with PKCE verifier)
        async with oauth_provider:
            token_info = await oauth_provider.exchange_code_for_tokens(
                code=request_body.code,
                auth_state=auth_state,
            )

            # Get user profile from Microsoft Graph (with ID token fallback)
            user_profile = await oauth_provider.get_user_profile(
                token_info.access_token,
                id_token=token_info.id_token
            )

        # Create session with user info and tokens
        settings = get_settings()
        session = await session_manager.create_session(
            user=user_profile,
            token_info=token_info,
        )

        # Set default organization and project from settings if available
        if settings.azure_devops.default_org_url:
            session.selected_org = settings.azure_devops.default_org_url
            logger.info(
                f"Setting default organization: {settings.azure_devops.default_org_url}",
                extra={"session_id": session.session_id[:16] + "..."},
            )
        if settings.azure_devops.default_project:
            session.selected_project = settings.azure_devops.default_project
            logger.info(
                f"Setting default project: {settings.azure_devops.default_project}",
                extra={"session_id": session.session_id[:16] + "..."},
            )

        # Update session to persist org/project selection
        if session.selected_org or session.selected_project:
            await session_manager.update_session(session)
            logger.info(
                "Session updated with default organization and project",
                extra={
                    "session_id": session.session_id[:16] + "...",
                    "org": session.selected_org,
                    "project": session.selected_project,
                },
            )

        # Create response with session info
        response = JSONResponse(
            content={
                "session_id": session.session_id,
                "user_id": session.user.id,
                "expires_at": (
                    session.token_info.expires_at
                    if session.token_info
                    else datetime.now(timezone.utc)
                    + timedelta(hours=settings.session_expiry_hours)
                ).isoformat(),
            }
        )

        # Set httpOnly cookie for session ID (secure, sameSite)
        # CRITICAL: samesite="none" is required for cross-domain cookies
        # secure=True is required when samesite="none" (browser requirement)
        settings = get_settings()
        response.set_cookie(
            key="session_id",
            value=session.session_id,
            httponly=True,
            secure=True,  # Required for samesite=none to work (browser enforces this)
            samesite="none",  # Required for cross-domain requests between frontend/backend
            max_age=settings.session_expiry_hours * 3600,
            path="/",
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication callback failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed. Please try again.",
        )


# =============================================================================
# Session Endpoints
# =============================================================================


@router.get("/session", tags=["session"])
async def get_session(
    session: SessionContext = Depends(get_current_session),
) -> dict[str, Any]:
    """Get current session info."""
    return {
        "session_id": session.session_id,
        "user_id": session.user.id,
        "user_name": session.user.display_name,
        "user_email": session.user.email,
        "organization_url": session.selected_org,
        "project_id": session.selected_project,
        "created_at": session.created_at.isoformat(),
        "last_accessed": session.last_accessed.isoformat(),
    }


@router.delete("/session", tags=["session"])
async def logout(
    request: Request,
    session: SessionContext = Depends(get_current_session),
) -> dict[str, str]:
    """Logout and destroy session."""
    session_manager: SessionManager = request.app.state.session_manager
    await session_manager.revoke_session(session.session_id)
    return {"message": "Logged out successfully"}


# =============================================================================
# Tool Endpoints
# =============================================================================


@router.get("/tools", tags=["tools"])
async def list_tools(request: Request) -> list[dict[str, Any]]:
    """List all available MCP tools."""
    session_manager: SessionManager = request.app.state.session_manager
    tools = create_all_tools(session_manager)

    return [
        {
            "name": tool.definition.name,
            "description": tool.definition.description,
            "category": tool.definition.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in tool.definition.parameters
            ],
        }
        for tool in tools
    ]


@router.post("/tools/execute", response_model=ToolExecuteResponse, tags=["tools"])
@limiter.limit(DEFAULT_RATE_LIMITS.tool_execution, key_func=rate_limit_key_func)
async def execute_tool(
    request: Request,
    request_body: ToolExecuteRequest,
    session: SessionContext = Depends(get_current_session),
) -> ToolExecuteResponse:
    """Execute an MCP tool."""
    session_manager: SessionManager = request.app.state.session_manager
    tools = create_all_tools(session_manager)

    # Find tool
    tool = next(
        (t for t in tools if t.definition.name == request_body.tool_name),
        None,
    )

    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool not found: {request_body.tool_name}",
        )

    # SSRF prevention: validate organization URL before passing to tool context
    if session.selected_org:
        parsed_org = urlparse(session.selected_org)
        if parsed_org.scheme != "https" or not (
            (parsed_org.hostname or "").endswith("dev.azure.com")
            or (parsed_org.hostname or "").endswith("visualstudio.com")
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid organization URL",
            )

    # Build context (conversation_id from request so draft changes can be persisted to that conversation)
    context = ToolContext(
        session_id=session.session_id,
        user_id=session.user.id,
        organization_url=session.selected_org,
        project_id=session.selected_project,
        project_name=session.metadata.get("project_name") or session.selected_project,
        access_token=(
            session.token_info.access_token if session.token_info else ""
        ),
        conversation_id=request_body.conversation_id,
    )

    try:
        # Handle 'context' parameter conflict: tool parameter 'context' (string)
        # conflicts with execute()'s 'context' argument (ToolContext)
        # Create a copy of parameters and rename 'context' to avoid conflict
        tool_params = dict(request_body.parameters)
        if 'context' in tool_params:
            # Rename 'context' parameter to 'additional_context' to avoid conflict
            tool_params['additional_context'] = tool_params.pop('context')

        result = await tool.execute(context, **tool_params)

        # Update session if tool returned context updates
        if result.metadata.get("update_context"):
            result.metadata["update_context"]
            await session_manager.update_session(session)

        return ToolExecuteResponse(
            status=result.status.value,
            data=result.data,
            message=result.message,
            error=result.error,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception:
        logger.exception("Tool execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tool execution failed. Please try again.",
        )


# =============================================================================
# Draft Endpoints (convenience endpoints)
# =============================================================================


@router.get("/drafts", tags=["drafts"])
async def list_drafts(
    request: Request,
    session: SessionContext = Depends(get_current_session),
) -> list[dict[str, Any]]:
    """List all draft work items."""
    session_manager: SessionManager = request.app.state.session_manager
    draft_collection = await session_manager.get_drafts(session.session_id)

    if not draft_collection:
        return []

    return [
        {
            "id": d.id,
            "type": d.work_item_type,
            "title": d.title,
            "created_at": d.created_at.isoformat(),
        }
        for d in draft_collection.items
    ]


@router.delete("/drafts", tags=["drafts"])
async def clear_drafts(
    request: Request,
    session: SessionContext = Depends(get_current_session),
) -> dict[str, str]:
    """Clear all draft work items."""
    session_manager: SessionManager = request.app.state.session_manager
    await session_manager.delete_drafts(session.session_id)
    return {"message": "All drafts cleared"}


# =============================================================================
# File Upload Endpoints
# =============================================================================


@router.post("/files/upload", tags=["files"])
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session: SessionContext = Depends(get_current_session),
) -> dict[str, Any]:
    """
    Upload a file and extract text content for use in work item generation.

    Supported formats:
    - Text files: .txt, .md
    - Word documents: .docx, .doc
    - PDF files: .pdf
    """
    # Validate file size (max 10MB)
    max_file_size = 10 * 1024 * 1024  # 10MB

    # L6: Check Content-Length before reading the full file body
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_file_size:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File size exceeds maximum limit of {max_file_size // (1024*1024)}MB",
                )
        except ValueError:
            pass  # Malformed Content-Length header, proceed and check after read

    # M4: Validate MIME type before processing
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Supported formats: txt, md, docx, doc, pdf.",
        )

    # Read file content
    file_content = await file.read()

    if len(file_content) > max_file_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum limit of {max_file_size / (1024*1024)}MB",
        )

    # Process file
    try:
        processor = FileProcessor()
        result = await processor.process_file(file_content, file.filename or "unknown")

        # Store file content in session for later use
        session_manager: SessionManager = request.app.state.session_manager
        await session_manager.store_file_content(
            session_id=session.session_id,
            file_id=result["file_name"],
            content=result["content"],
        )

        return {
            "file_name": result["file_name"],
            "content": result["content"],
            "file_type": result["file_type"],
            "message": f"File '{result['file_name']}' processed successfully",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process file. Ensure the file format is supported (txt, md, docx, pdf).",
        ) from e
