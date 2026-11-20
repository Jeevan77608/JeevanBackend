"""
FastAPI authentication dependencies.

Provides dependency injection for:
- Session validation
- Token validation and refresh
- User context extraction
- OAuth provider access
- Mock session for development when auth is disabled
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Cookie, Depends, Header, HTTPException, Request, status

from src.auth.oauth import AzureADOAuthProvider
from src.config.settings import get_settings
from src.models.user import SessionContext, TokenInfo, UserProfile
from src.session.manager import (
    SessionError,
    SessionExpiredError,
    SessionFingerprintMismatchError,
    SessionManager,
    SessionNotFoundError,
)
from src.session.models import SessionFingerprint

logger = logging.getLogger(__name__)

# In-memory locks for token refresh (keyed by session_id) to prevent races
_token_refresh_locks: dict[str, asyncio.Lock] = {}


# =============================================================================
# Mock Session for Development (when AUTH_ENABLED=false)
# =============================================================================


def create_mock_session() -> SessionContext:
    """
    Create a mock session for development when authentication is disabled.

    Uses Personal Access Token (PAT) from settings if available.
    WARNING: This should only be used for local development!
    """
    settings = get_settings()

    # Use PAT from settings if available
    access_token = "dev-access-token-no-auth"
    if settings.azure_devops.pat:
        access_token = settings.azure_devops.pat.get_secret_value()

    return SessionContext(
        session_id="dev-session-no-auth",
        user=UserProfile(
            id="dev-user-001",
            display_name="Development User",
            email="dev@localhost",
            user_principal_name="dev@localhost",
            tenant_id="dev-tenant",
        ),
        token_info=TokenInfo(
            access_token=access_token,
            refresh_token=None,
            token_type="Basic",  # PAT uses Basic auth
            expires_at=datetime.now(timezone.utc),
            scope="user_impersonation",
        ),
        created_at=datetime.now(timezone.utc),
        last_accessed=datetime.now(timezone.utc),
        selected_org=settings.azure_devops.default_org_url,
        selected_project=settings.azure_devops.default_project,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def extract_fingerprint(
    user_agent: str | None = None,
    client_ip: str | None = None,
    accept_language: str | None = None,
) -> SessionFingerprint | None:
    """Extract session fingerprint from request headers."""
    if not user_agent:
        return None
    return SessionFingerprint.create(
        user_agent=user_agent,
        client_ip=client_ip,
        accept_language=accept_language,
    )


def get_client_ip(request: Request) -> str | None:
    """Get client IP from request, considering proxies."""
    # Check X-Forwarded-For header (set by reverse proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(",")[0].strip()

    # Fall back to direct client IP
    if request.client:
        return request.client.host
    return None


# =============================================================================
# Dependencies
# =============================================================================


async def get_session_manager(request: Request) -> SessionManager:
    """Get session manager from application state."""
    session_manager = getattr(request.app.state, "session_manager", None)
    if not session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session manager not available",
        )
    return session_manager


async def get_oauth_provider(request: Request) -> AzureADOAuthProvider | None:
    """
    Get OAuth provider from application state.

    Returns None when authentication is disabled.
    """
    settings = get_settings()

    # OAuth provider not available when auth is disabled
    if not settings.auth_enabled:
        return None

    oauth_provider = getattr(request.app.state, "oauth_provider", None)
    if not oauth_provider:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth provider not available",
        )
    return oauth_provider


def _validate_internal_key(request: Request) -> bool:
    """Validate X-Internal-Key header for server-to-server requests."""
    settings = get_settings()
    expected_key = settings.internal_api_key.get_secret_value()
    if not expected_key:
        # No internal key configured (development) — allow without key
        return True
    provided_key = request.headers.get("X-Internal-Key", "")
    if not provided_key:
        return False
    # Use hmac.compare_digest for timing-safe comparison
    import hmac
    return hmac.compare_digest(provided_key, expected_key)


async def get_optional_session(
    request: Request,
    session_id: Annotated[str | None, Cookie(alias="session_id")] = None,
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    user_agent: Annotated[str | None, Header(alias="User-Agent")] = None,
    accept_language: Annotated[str | None, Header(alias="Accept-Language")] = None,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionContext | None:
    """
    Get session if available (doesn't require authentication).

    Returns None if no valid session exists.
    """
    # Determine request origin:
    # - X-Session-ID + X-Internal-Key = server-to-server (Next.js proxy) → skip fingerprint
    # - X-Session-ID without X-Internal-Key = browser direct → do fingerprint validation
    # - Cookie only = browser via cookie → do fingerprint validation
    has_internal_key = _validate_internal_key(request)
    from_proxy = x_session_id is not None and has_internal_key
    session_id = x_session_id or session_id
    if not session_id:
        return None

    try:
        # Skip fingerprint validation for server-to-server proxy requests
        # (X-Session-ID + valid X-Internal-Key) because the Next.js server
        # has a different User-Agent than the browser
        fingerprint = None
        if not from_proxy:
            client_ip = get_client_ip(request)
            fingerprint = extract_fingerprint(user_agent, client_ip, accept_language)

        # Get and validate session
        session = await session_manager.get_session(
            session_id,
            fingerprint=fingerprint,
            extend_expiry=True,
        )
        return session

    except SessionError as e:
        logger.debug(f"Optional session not available: {e}")
        return None


async def get_current_session(
    request: Request,
    session_id: Annotated[str | None, Cookie(alias="session_id")] = None,
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    user_agent: Annotated[str | None, Header(alias="User-Agent")] = None,
    accept_language: Annotated[str | None, Header(alias="Accept-Language")] = None,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionContext:
    """
    Get current authenticated session (requires authentication).

    Accepts session_id from Cookie or X-Session-ID header.
    When AUTH_ENABLED=false, returns a mock session for development.
    Raises HTTPException if session is invalid or expired.
    """
    settings = get_settings()

    # Return mock session when auth is disabled (development only)
    if not settings.auth_enabled:
        logger.debug("Authentication disabled - returning mock session")
        return create_mock_session()

    # Determine request origin:
    # - X-Session-ID + X-Internal-Key = server-to-server (Next.js proxy) → skip fingerprint
    # - X-Session-ID without X-Internal-Key = browser direct → do fingerprint validation
    # - Cookie only = browser via cookie → do fingerprint validation
    has_internal_key = _validate_internal_key(request)
    from_proxy = x_session_id is not None and has_internal_key
    session_id = x_session_id or session_id
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Skip fingerprint validation for server-to-server proxy requests
        # (X-Session-ID + valid X-Internal-Key) because the Next.js server
        # has a different User-Agent than the browser
        fingerprint = None
        if not from_proxy:
            client_ip = get_client_ip(request)
            fingerprint = extract_fingerprint(user_agent, client_ip, accept_language)

        # Get and validate session
        session = await session_manager.get_session(
            session_id,
            fingerprint=fingerprint,
            extend_expiry=True,
        )

        # Automatically refresh access token if expired or about to expire
        if settings.auth_enabled and settings.auth_provider == "azure_ad" and session.token_info:
            oauth_provider = getattr(request.app.state, "oauth_provider", None)
            if oauth_provider:
                from src.auth.oauth import TokenValidator
                validator = TokenValidator(oauth_provider)

                # Check if token needs refresh (expired or expiring soon)
                if validator.needs_refresh(session.token_info) or not validator.is_token_valid(session.token_info):
                    # Use per-session lock to prevent concurrent refresh races
                    lock = _token_refresh_locks.setdefault(session.session_id, asyncio.Lock())
                    if lock.locked():
                        # Another request is already refreshing — skip and use current token
                        logger.debug("Token refresh already in progress, skipping")
                    else:
                        try:
                            async with lock:
                                # Refresh the token
                                async with oauth_provider:
                                    new_token_info = await validator.ensure_valid_token(session.token_info)

                                # Update session with new token
                                session.token_info = new_token_info
                                await session_manager.update_session(session, fingerprint=fingerprint)

                                logger.debug("Access token refreshed automatically")
                        except Exception as e:
                            logger.warning(f"Failed to refresh token: {e}")
                            # If refresh fails, session will still be returned
                            # The API call will fail with 401, prompting re-authentication

        return session

    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except SessionExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except SessionFingerprintMismatchError:
        logger.warning(
            "Session fingerprint mismatch detected",
            extra={"session_id": session_id[:16] + "..."},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session security validation failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except SessionError as e:
        logger.error(f"Session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
