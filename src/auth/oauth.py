"""
Azure AD OAuth 2.0 authentication with PKCE support.

This module implements secure OAuth 2.0 authentication following:
- RFC 6749 (OAuth 2.0)
- RFC 7636 (PKCE - Proof Key for Code Exchange)
- Microsoft Identity Platform best practices

Security Features:
- PKCE to prevent authorization code interception
- State parameter for CSRF protection
- Secure token storage
- Automatic token refresh
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Final
from urllib.parse import urlencode

import httpx
from jose import JWTError
from jose import jwt as jose_jwt
from msal import ConfidentialClientApplication

from src.config.settings import Settings
from src.models.user import TokenInfo, UserProfile

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class AuthenticationError(Exception):
    """Base exception for authentication errors."""

    pass


class AuthorizationError(AuthenticationError):
    """Authorization failed (invalid code, state, etc.)."""

    pass


class TokenError(AuthenticationError):
    """Token operation failed (refresh, validation, etc.)."""

    pass


class UserInfoError(AuthenticationError):
    """Failed to retrieve user information."""

    pass


# =============================================================================
# Constants
# =============================================================================

# PKCE code verifier length (43-128 chars, we use 64 for security)
CODE_VERIFIER_LENGTH: Final[int] = 64

# Token refresh buffer (refresh before expiry)
TOKEN_REFRESH_BUFFER: Final[timedelta] = timedelta(minutes=5)


# =============================================================================
# PKCE Helper
# =============================================================================


@dataclass(frozen=True, slots=True)
class PKCEChallenge:
    """PKCE challenge pair for OAuth 2.0 authorization."""

    code_verifier: str
    code_challenge: str
    code_challenge_method: str = "S256"

    @classmethod
    def generate(cls) -> PKCEChallenge:
        """
        Generate a new PKCE challenge pair.

        Uses SHA-256 (S256) method as recommended by RFC 7636.
        """
        # Generate random verifier (URL-safe base64, 64 bytes)
        code_verifier = secrets.token_urlsafe(CODE_VERIFIER_LENGTH)

        # Create SHA-256 hash of verifier
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()

        # Base64 URL encode without padding
        code_challenge = (
            base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        )

        return cls(
            code_verifier=code_verifier,
            code_challenge=code_challenge,
        )


# =============================================================================
# Auth State
# =============================================================================


@dataclass
class AuthorizationState:
    """State for tracking OAuth authorization flow."""

    state: str
    pkce: PKCEChallenge
    redirect_uri: str
    created_at: datetime

    @classmethod
    def create(cls, redirect_uri: str) -> AuthorizationState:
        """Create a new authorization state."""
        return cls(
            state=secrets.token_urlsafe(32),
            pkce=PKCEChallenge.generate(),
            redirect_uri=redirect_uri,
            created_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "state": self.state,
            "code_verifier": self.pkce.code_verifier,
            "code_challenge": self.pkce.code_challenge,
            "redirect_uri": self.redirect_uri,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthorizationState:
        """Create from dictionary."""
        return cls(
            state=data["state"],
            pkce=PKCEChallenge(
                code_verifier=data["code_verifier"],
                code_challenge=data["code_challenge"],
            ),
            redirect_uri=data["redirect_uri"],
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        )


# =============================================================================
# Azure AD OAuth Provider
# =============================================================================


class AzureADOAuthProvider:
    """
    Azure AD OAuth 2.0 provider with PKCE support.

    Handles the complete OAuth flow:
    1. Generate authorization URL with PKCE
    2. Exchange authorization code for tokens
    3. Refresh expired tokens
    4. Retrieve user profile from Microsoft Graph

    Thread Safety:
    - This class is thread-safe. Each operation is stateless.
    - State management should be handled by SessionManager.
    """

    # Microsoft Graph API endpoint for user profile
    GRAPH_API_URL: Final[str] = "https://graph.microsoft.com/v1.0"

    def __init__(self, settings: Settings) -> None:
        """
        Initialize OAuth provider.

        Args:
            settings: Application settings
        """
        self._settings = settings
        self._ad_settings = settings.azure_ad

        # Initialize MSAL confidential client
        self._msal_app = ConfidentialClientApplication(
            client_id=self._ad_settings.client_id,
            client_credential=self._ad_settings.client_secret.get_secret_value(),
            authority=self._ad_settings.authority,
        )

        # HTTP client for Graph API calls
        self._http_client: httpx.AsyncClient | None = None

        # JWKS cache for ID token verification
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_uri = f"https://login.microsoftonline.com/{self._ad_settings.tenant_id}/discovery/v2.0/keys"

        logger.info(
            "Azure AD OAuth provider initialized",
            extra={"client_id": self._ad_settings.client_id[:8] + "..."},
        )

    async def __aenter__(self) -> AzureADOAuthProvider:
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client, ensuring it's initialized."""
        if self._http_client is None:
            raise RuntimeError("OAuth provider not initialized as async context manager")
        return self._http_client

    # =========================================================================
    # Authorization URL Generation
    # =========================================================================

    def create_authorization_state(self, redirect_uri: str | None = None) -> AuthorizationState:
        """
        Create a new authorization state for OAuth flow.

        Args:
            redirect_uri: Optional redirect URI. If not provided, uses the default from settings.

        Returns:
            AuthorizationState with PKCE challenge
        """
        uri = redirect_uri or self._ad_settings.redirect_uri
        logger.info(f"Using redirect_uri: {uri}", extra={"redirect_uri": uri})
        return AuthorizationState.create(uri)

    def get_authorization_url(self, auth_state: AuthorizationState) -> str:
        """
        Generate OAuth 2.0 authorization URL.

        Args:
            auth_state: Authorization state with PKCE

        Returns:
            Authorization URL for user redirect
        """
        # Build authorization URL manually to ensure PKCE challenge is used exactly as generated
        params = {
            "client_id": self._ad_settings.client_id,
            "response_type": "code",
            "redirect_uri": auth_state.redirect_uri,
            "response_mode": "query",
            "scope": " ".join(self._ad_settings.scopes),
            "state": auth_state.state,
            "code_challenge": auth_state.pkce.code_challenge,
            "code_challenge_method": auth_state.pkce.code_challenge_method,
        }

        auth_url = f"{self._ad_settings.authorize_endpoint}?{urlencode(params)}"

        scopes_list = self._ad_settings.scopes
        logger.info(
            f"Generated authorization URL with scopes: {scopes_list}. "
            f"State: {auth_state.state[:16]}..., "
            f"Redirect URI: {auth_state.redirect_uri}"
        )

        return auth_url

    # =========================================================================
    # Token Exchange
    # =========================================================================

    async def exchange_code_for_tokens(
        self,
        code: str,
        auth_state: AuthorizationState,
    ) -> TokenInfo:
        """
        Exchange authorization code for access and refresh tokens.

        Args:
            code: Authorization code from callback
            auth_state: Original authorization state with PKCE verifier

        Returns:
            TokenInfo with access and refresh tokens

        Raises:
            AuthorizationError: If token exchange fails
        """
        try:
            # MSAL ConfidentialClientApplication doesn't support code_verifier parameter
            # We need to manually make the token request with PKCE
            http_client = self._get_http_client()

            # Validate PKCE verifier matches challenge
            verifier_bytes = auth_state.pkce.code_verifier.encode("ascii")
            challenge_hash = hashlib.sha256(verifier_bytes).digest()
            expected_challenge = base64.urlsafe_b64encode(challenge_hash).decode("ascii").rstrip("=")

            if expected_challenge != auth_state.pkce.code_challenge:
                logger.error(
                    f"PKCE mismatch! Expected challenge: {expected_challenge[:16]}..., "
                    f"Got: {auth_state.pkce.code_challenge[:16]}..."
                )
                raise AuthorizationError("PKCE code_verifier does not match code_challenge")

            logger.debug(
                f"PKCE validation passed. "
                f"Redirect URI: {auth_state.redirect_uri}"
            )

            # Prepare token request with PKCE
            scopes_str = " ".join(self._ad_settings.scopes)
            logger.debug(f"Token request scopes: {scopes_str}")

            token_data = {
                "grant_type": "authorization_code",
                "client_id": self._ad_settings.client_id,
                "client_secret": self._ad_settings.client_secret.get_secret_value(),
                "code": code,
                "redirect_uri": auth_state.redirect_uri,
                "code_verifier": auth_state.pkce.code_verifier,  # PKCE verifier
                "scope": scopes_str,
            }

            # Make token request
            response = await http_client.post(
                self._ad_settings.token_endpoint,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                error_text = await response.aread()
                logger.warning(
                    f"Token exchange failed: {response.status_code} - {error_text.decode()}",
                )
                raise AuthorizationError(
                    f"Token exchange failed: {response.status_code} - {error_text.decode()}"
                )

            result = response.json()

            if "error" in result:
                error_desc = result.get("error_description", result.get("error"))
                logger.warning(
                    f"Token exchange failed: {error_desc}",
                    extra={"error": result.get("error")},
                )
                raise AuthorizationError(f"Token exchange failed: {error_desc}")

            # Parse token response
            token_info = self._parse_token_response(result)

            logger.info(
                "Token exchange successful",
                extra={
                    "expires_at": token_info.expires_at.isoformat(),
                    "has_id_token": bool(token_info.id_token),
                    "has_refresh_token": bool(token_info.refresh_token),
                },
            )

            return token_info

        except AuthorizationError:
            raise
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            raise AuthorizationError(f"Token exchange failed: {e}") from e

    def _parse_token_response(self, result: dict[str, Any]) -> TokenInfo:
        """Parse MSAL token response into TokenInfo."""
        # Calculate expiration time
        expires_in = result.get("expires_in", 3600)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        # Log token response details for debugging
        id_token = result.get("id_token")
        if id_token:
            logger.info("ID token received in token response")
        else:
            logger.error(
                "No ID token in token response! "
                "This means the OAuth request did not include 'openid' scope. "
                "Check that .default scope includes openid."
            )

        logger.debug(f"Token response keys: {list(result.keys())}")
        logger.debug(f"Scopes in response: {result.get('scope', 'Not provided')}")
        logger.debug(f"Token type: {result.get('token_type', 'Not provided')}")

        return TokenInfo(
            access_token=result["access_token"],
            refresh_token=result.get("refresh_token"),
            token_type=result.get("token_type", "Bearer"),
            expires_at=expires_at,
            scope=" ".join(result.get("scope", [])) if isinstance(result.get("scope"), list) else result.get("scope", ""),
            id_token=id_token,  # Store ID token for user profile fallback
        )

    # =========================================================================
    # Token Refresh
    # =========================================================================

    async def refresh_tokens(self, refresh_token: str) -> TokenInfo:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Current refresh token

        Returns:
            New TokenInfo with fresh tokens

        Raises:
            TokenError: If refresh fails
        """
        try:
            # Get accounts from cache (MSAL requirement)
            accounts = self._msal_app.get_accounts()

            if accounts:
                # Try silent token acquisition
                result = self._msal_app.acquire_token_silent(
                    scopes=self._ad_settings.scopes,
                    account=accounts[0],
                )
            else:
                # Fall back to refresh token grant
                result = self._msal_app.acquire_token_by_refresh_token(
                    refresh_token=refresh_token,
                    scopes=self._ad_settings.scopes,
                )

            if not result or "error" in result:
                error_desc = result.get("error_description", "Unknown error") if result else "No result"
                raise TokenError(f"Token refresh failed: {error_desc}")

            token_info = self._parse_token_response(result)

            logger.info(
                "Token refresh successful",
                extra={"expires_at": token_info.expires_at.isoformat()},
            )

            return token_info

        except TokenError:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            raise TokenError(f"Token refresh failed: {e}") from e

    def should_refresh_token(self, token_info: TokenInfo) -> bool:
        """Check if token should be refreshed (approaching expiry)."""
        if not token_info.refresh_token:
            return False
        buffer_time = datetime.now(timezone.utc) + TOKEN_REFRESH_BUFFER
        return token_info.expires_at <= buffer_time

    # =========================================================================
    # User Profile
    # =========================================================================

    async def get_user_profile(self, access_token: str, id_token: str | None = None) -> UserProfile:
        """
        Retrieve user profile from ID token.

        The ID token contains all user information we need (name, email, user ID, etc.)
        when using .default scope which includes openid and profile scopes.

        Args:
            access_token: Valid access token (not used, kept for API compatibility)
            id_token: ID token containing user information

        Returns:
            UserProfile with user information

        Raises:
            UserInfoError: If profile retrieval fails
        """
        if not id_token:
            logger.error("No ID token provided - cannot retrieve user profile")
            raise UserInfoError(
                "ID token is required for user profile. "
                "Ensure the OAuth request includes 'openid' and 'profile' scopes."
            )

        try:
            logger.info(f"Parsing user profile from ID token (length: {len(id_token)})")
            profile = await self._parse_id_token(id_token)
            logger.info(f"Successfully parsed user profile: {profile.display_name}")
            return profile
        except Exception as e:
            logger.error(f"Failed to parse ID token: {e}", exc_info=True)
            raise UserInfoError(
                f"Failed to parse ID token and retrieve user profile: {e}. "
                "Please check that the OAuth request includes 'openid' and 'profile' scopes."
            ) from e

    async def _fetch_jwks(self) -> dict[str, Any]:
        """Fetch and cache Azure AD JWKS keys for JWT verification."""
        if self._jwks_cache is not None:
            return self._jwks_cache

        client = self._http_client or httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        try:
            response = await client.get(self._jwks_uri)
            response.raise_for_status()
            self._jwks_cache = response.json()
            logger.info("Fetched Azure AD JWKS keys for ID token verification")
            return self._jwks_cache
        except Exception as e:
            logger.warning(f"Failed to fetch JWKS keys: {e}")
            raise
        finally:
            if self._http_client is None:
                await client.aclose()

    async def _parse_id_token(self, id_token: str) -> UserProfile:
        """
        Parse and verify ID token JWT signature against Azure AD JWKS.

        In production, the token signature MUST be verified. In development,
        falls back to unsigned decode with a warning if JWKS fetch fails.

        Args:
            id_token: JWT ID token

        Returns:
            UserProfile with user information

        Raises:
            UserInfoError: If token parsing or verification fails
        """
        try:
            if not id_token or not id_token.strip():
                raise UserInfoError("ID token is empty or None")

            claims: dict[str, Any] | None = None
            is_production = self._settings.is_production

            # Attempt signature-verified decode
            try:
                jwks = await self._fetch_jwks()
                claims = jose_jwt.decode(
                    id_token,
                    jwks,
                    algorithms=["RS256"],
                    audience=self._ad_settings.client_id,
                    issuer=f"https://login.microsoftonline.com/{self._ad_settings.tenant_id}/v2.0",
                    options={"verify_exp": True},
                )
                logger.info("ID token JWT signature verified successfully")
            except JWTError as e:
                if is_production:
                    logger.error(f"ID token JWT verification failed: {e}")
                    raise UserInfoError(
                        "ID token signature verification failed. Token may be tampered."
                    ) from e
                logger.warning(
                    f"JWT verification failed in development, falling back to unsigned decode: {e}"
                )
            except Exception as e:
                if is_production:
                    logger.error(f"JWKS fetch/verification error: {e}")
                    raise UserInfoError(
                        "Unable to verify ID token signature."
                    ) from e
                logger.warning(
                    f"JWKS unavailable in development, falling back to unsigned decode: {e}"
                )

            # Fallback: unsigned decode (development only — guarded above)
            if claims is None:
                parts = id_token.split(".")
                if len(parts) != 3:
                    raise UserInfoError(f"Invalid ID token format: expected 3 parts, got {len(parts)}")
                payload = parts[1]
                padding = 4 - len(payload) % 4
                if padding != 4:
                    payload += "=" * padding
                decoded = base64.urlsafe_b64decode(payload)
                claims = json.loads(decoded)

            # Extract user info from claims
            user_id = claims.get("oid") or claims.get("sub", "")
            if not user_id:
                raise UserInfoError("No user ID (oid or sub) found in ID token claims")

            display_name = (
                claims.get("name")
                or claims.get("display_name")
                or claims.get("given_name")
                or claims.get("preferred_username")
                or "Unknown"
            )

            email = (
                claims.get("email")
                or claims.get("upn")
                or claims.get("preferred_username")
                or claims.get("unique_name")
                or ""
            )

            user_principal_name = (
                claims.get("preferred_username")
                or claims.get("upn")
                or email
                or ""
            )

            tenant_id = claims.get("tid") or self._ad_settings.tenant_id

            return UserProfile(
                id=user_id,
                display_name=display_name,
                email=email,
                user_principal_name=user_principal_name,
                tenant_id=tenant_id,
            )

        except UserInfoError:
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode ID token JSON: {e}")
            raise UserInfoError(f"Failed to decode ID token JSON: {e}") from e
        except Exception as e:
            logger.error(f"Failed to parse ID token: {e}", exc_info=True)
            raise UserInfoError(f"Failed to parse ID token: {e}") from e


# =============================================================================
# Token Validator
# =============================================================================


class TokenValidator:
    """
    Validates and manages OAuth tokens.

    Provides middleware-friendly validation for API requests.
    """

    def __init__(self, oauth_provider: AzureADOAuthProvider) -> None:
        """Initialize token validator."""
        self._oauth = oauth_provider

    def is_token_valid(self, token_info: TokenInfo) -> bool:
        """Check if token is still valid (not expired)."""
        return datetime.now(timezone.utc) < token_info.expires_at

    def needs_refresh(self, token_info: TokenInfo) -> bool:
        """Check if token needs refresh."""
        return self._oauth.should_refresh_token(token_info)

    async def ensure_valid_token(self, token_info: TokenInfo) -> TokenInfo:
        """
        Ensure token is valid, refreshing if needed.

        Args:
            token_info: Current token info

        Returns:
            Valid token info (refreshed if needed)

        Raises:
            TokenError: If token is invalid and cannot be refreshed
        """
        if self.is_token_valid(token_info) and not self.needs_refresh(token_info):
            return token_info

        if not token_info.refresh_token:
            raise TokenError("Token expired and no refresh token available")

        return await self._oauth.refresh_tokens(token_info.refresh_token)
