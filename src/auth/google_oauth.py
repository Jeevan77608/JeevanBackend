"""Google OpenID Connect authentication."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from jose import JWTError
from jose.exceptions import JWTClaimsError
from jose import jwt

from src.config.settings import Settings
from src.models.user import TokenInfo, UserProfile
from src.auth.oauth import AuthorizationState

logger = logging.getLogger(__name__)


class GoogleOAuthError(Exception):
    """Raised when Google OAuth authentication fails."""


class GoogleOAuthProvider:
    """OAuth 2.0 authorization-code flow with ID-token signature validation."""

    AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
    CERTS_ENDPOINT = "https://www.googleapis.com/oauth2/v3/certs"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client_id = settings.google_client_id
        self._client_secret = settings.google_client_secret.get_secret_value()
        self._redirect_uri = settings.google_redirect_uri
        self._certs: dict[str, Any] | None = None

    def create_authorization_state(self) -> AuthorizationState:
        return AuthorizationState.create(self._redirect_uri)

    def get_authorization_url(self, auth_state: AuthorizationState) -> str:
        return f"{self.AUTHORIZATION_ENDPOINT}?{urlencode({
            'client_id': self._client_id,
            'redirect_uri': auth_state.redirect_uri,
            'response_type': 'code',
            'scope': 'openid email profile',
            'state': auth_state.state,
            'code_challenge': auth_state.pkce.code_challenge,
            'code_challenge_method': auth_state.pkce.code_challenge_method,
        })}"

    async def exchange_code_for_user(
        self, code: str, auth_state: AuthorizationState
    ) -> tuple[UserProfile, TokenInfo]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                self.TOKEN_ENDPOINT,
                data={
                    "code": code,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "redirect_uri": auth_state.redirect_uri,
                    "grant_type": "authorization_code",
                    "code_verifier": auth_state.pkce.code_verifier,
                },
            )
            if response.is_error:
                logger.warning("Google token exchange rejected with HTTP %s", response.status_code)
                raise GoogleOAuthError("Google sign-in could not be completed.")
            token_data = response.json()

            id_token = token_data.get("id_token")
            if not isinstance(id_token, str):
                raise GoogleOAuthError("Google did not return an ID token.")

            access_token = token_data.get("access_token")
            if not isinstance(access_token, str):
                raise GoogleOAuthError("Google did not return an access token.")

            claims = await self._verify_id_token(client, id_token, access_token)
            email = claims.get("email")
            subject = claims.get("sub")
            if (
                not isinstance(email, str)
                or not isinstance(subject, str)
                or claims.get("email_verified") is not True
            ):
                raise GoogleOAuthError("A verified Google email address is required.")

        expires_in = int(token_data.get("expires_in", 3600))
        token_info = TokenInfo(
            access_token=access_token,
            refresh_token=token_data.get("refresh_token"),
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            scope=token_data.get("scope", "openid email profile"),
            id_token=id_token,
        )
        display_name = claims.get("name")
        return (
            UserProfile(
                id=f"google:{subject}",
                display_name=display_name if isinstance(display_name, str) else email,
                email=email,
                user_principal_name=email,
                tenant_id="google",
            ),
            token_info,
        )

    async def _verify_id_token(
        self, client: httpx.AsyncClient, id_token: str, access_token: str
    ) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(id_token)
            key_id = header.get("kid")
            if not isinstance(key_id, str) or header.get("alg") != "RS256":
                raise GoogleOAuthError("Google returned an invalid ID token.")

            if self._certs is None:
                response = await client.get(self.CERTS_ENDPOINT)
                response.raise_for_status()
                self._certs = {
                    key["kid"]: key
                    for key in response.json().get("keys", [])
                    if isinstance(key, dict) and isinstance(key.get("kid"), str)
                }

            key = self._certs.get(key_id)
            if key is None:
                self._certs = None
                response = await client.get(self.CERTS_ENDPOINT)
                response.raise_for_status()
                self._certs = {
                    item["kid"]: item
                    for item in response.json().get("keys", [])
                    if isinstance(item, dict) and isinstance(item.get("kid"), str)
                }
                key = self._certs.get(key_id)
            if key is None:
                raise GoogleOAuthError("Google signing key was not found.")

            claims = jwt.decode(
                id_token,
                key,
                algorithms=["RS256"],
                audience=self._client_id,
                access_token=access_token,
                options={"verify_iss": False},
            )
            if claims.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
                raise GoogleOAuthError("Google ID token issuer is invalid.")
            return claims
        except GoogleOAuthError:
            raise
        except JWTClaimsError as error:
            logger.warning("Google ID token claim validation failed: %s", error)
            raise GoogleOAuthError("Google ID token validation failed.") from error
        except (JWTError, httpx.HTTPError, ValueError, KeyError) as error:
            logger.warning("Google ID token verification failed: %s", type(error).__name__)
            raise GoogleOAuthError("Google ID token validation failed.") from error
