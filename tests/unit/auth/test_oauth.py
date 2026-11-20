"""Tests for src/auth/oauth.py — PKCE, AuthorizationState, OAuth provider, TokenValidator."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from src.auth.oauth import (
    AuthenticationError,
    AuthorizationError,
    AuthorizationState,
    AzureADOAuthProvider,
    PKCEChallenge,
    TokenError,
    TokenValidator,
    UserInfoError,
)
from src.models.user import TokenInfo, UserProfile


# =============================================================================
# PKCEChallenge
# =============================================================================


class TestPKCEChallenge:
    """Tests for the PKCE challenge generation."""

    def test_generate_produces_valid_challenge(self) -> None:
        """SHA-256 of the verifier, base64url-encoded without padding, must equal the challenge."""
        pkce = PKCEChallenge.generate()
        digest = hashlib.sha256(pkce.code_verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert pkce.code_challenge == expected

    def test_generate_code_challenge_method_is_s256(self) -> None:
        """The challenge method must always be S256."""
        pkce = PKCEChallenge.generate()
        assert pkce.code_challenge_method == "S256"

    def test_generate_produces_unique_values(self) -> None:
        """Two consecutive calls must produce different verifiers."""
        a = PKCEChallenge.generate()
        b = PKCEChallenge.generate()
        assert a.code_verifier != b.code_verifier
        assert a.code_challenge != b.code_challenge

    def test_frozen_dataclass(self) -> None:
        """PKCEChallenge instances must be immutable (frozen dataclass)."""
        pkce = PKCEChallenge.generate()
        with pytest.raises(AttributeError):
            pkce.code_verifier = "tampered"  # type: ignore[misc]


# =============================================================================
# AuthorizationState
# =============================================================================


class TestAuthorizationState:
    """Tests for OAuth authorization-state bookkeeping."""

    def test_create_generates_state_and_pkce(self) -> None:
        """create() must populate state string and a valid PKCEChallenge."""
        auth_state = AuthorizationState.create("https://example.com/callback")
        assert isinstance(auth_state.state, str)
        assert len(auth_state.state) > 0
        assert isinstance(auth_state.pkce, PKCEChallenge)
        assert auth_state.redirect_uri == "https://example.com/callback"

    def test_to_dict_serializable(self) -> None:
        """to_dict() must produce a dict with the expected keys."""
        auth_state = AuthorizationState.create("https://example.com/callback")
        d = auth_state.to_dict()
        expected_keys = {"state", "code_verifier", "code_challenge", "redirect_uri", "created_at"}
        assert expected_keys.issubset(set(d.keys()))
        # All values should be JSON-serializable strings or primitives
        json.dumps(d)  # must not raise

    def test_from_dict_round_trip(self) -> None:
        """from_dict(to_dict()) must reconstruct an equivalent AuthorizationState."""
        original = AuthorizationState.create("https://example.com/callback")
        d = original.to_dict()
        restored = AuthorizationState.from_dict(d)
        assert restored.state == original.state
        assert restored.pkce.code_verifier == original.pkce.code_verifier
        assert restored.pkce.code_challenge == original.pkce.code_challenge
        assert restored.redirect_uri == original.redirect_uri

    def test_from_dict_handles_z_suffix(self) -> None:
        """from_dict must handle datetime strings with trailing 'Z' (Python 3.10 compat)."""
        auth_state = AuthorizationState.create("https://example.com/callback")
        d = auth_state.to_dict()
        # Force the Z-suffix format that Pydantic v2 model_dump(mode="json") produces
        d["created_at"] = "2025-06-15T12:00:00Z"
        restored = AuthorizationState.from_dict(d)
        assert restored.created_at.tzinfo is not None


# =============================================================================
# Settings fixture for AzureADOAuthProvider
# =============================================================================


@pytest.fixture()
def auth_settings(monkeypatch):
    """Return a Settings instance with auth enabled and Azure AD configured."""
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
    from src.config.settings import Settings, clear_settings_cache

    clear_settings_cache()
    s = Settings()
    yield s
    clear_settings_cache()


@pytest.fixture()
def mock_msal():
    """Patch the MSAL ConfidentialClientApplication used by AzureADOAuthProvider."""
    with patch("src.auth.oauth.ConfidentialClientApplication") as mock_cls:
        mock_app = MagicMock()
        mock_cls.return_value = mock_app
        yield mock_app


@pytest.fixture()
def oauth_provider(auth_settings, mock_msal):
    """Return an AzureADOAuthProvider with mocked MSAL."""
    return AzureADOAuthProvider(auth_settings)


# =============================================================================
# AzureADOAuthProvider
# =============================================================================


class TestAzureADOAuthProvider:
    """Tests for the Azure AD OAuth provider."""

    def test_create_authorization_state_uses_default_redirect(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """When no redirect_uri is passed, the provider default is used."""
        auth_state = oauth_provider.create_authorization_state()
        assert auth_state.redirect_uri == "https://example.com/api/auth/callback"

    def test_create_authorization_state_custom_redirect(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """An explicit redirect_uri overrides the default."""
        auth_state = oauth_provider.create_authorization_state(
            redirect_uri="https://custom.example.com/cb"
        )
        assert auth_state.redirect_uri == "https://custom.example.com/cb"

    def test_get_authorization_url_contains_required_params(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """The authorization URL must include client_id, redirect_uri, state, and code_challenge."""
        auth_state = oauth_provider.create_authorization_state()
        url = oauth_provider.get_authorization_url(auth_state)
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        assert "client_id" in qs
        assert "redirect_uri" in qs
        assert "state" in qs
        assert qs["state"] == [auth_state.state]
        assert "code_challenge" in qs
        assert qs["code_challenge"] == [auth_state.pkce.code_challenge]
        assert "response_type" in qs
        assert qs["response_type"] == ["code"]

    async def test_exchange_code_for_tokens_success(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """A 200 response with valid JSON returns TokenInfo."""
        auth_state = oauth_provider.create_authorization_state()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "access-tok",
            "refresh_token": "refresh-tok",
            "expires_in": 3600,
            "scope": "openid profile",
            "id_token": "id-tok",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        async with oauth_provider as provider:
            provider._http_client = mock_client
            token_info = await provider.exchange_code_for_tokens("auth-code", auth_state)

        assert token_info.access_token == "access-tok"
        assert token_info.refresh_token == "refresh-tok"
        mock_client.post.assert_called_once()

    async def test_exchange_code_for_tokens_http_error(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """A non-200 response must raise AuthorizationError."""
        auth_state = oauth_provider.create_authorization_state()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.json.return_value = {"error": "invalid_grant"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with pytest.raises(AuthorizationError):
            async with oauth_provider as provider:
                provider._http_client = mock_client
                await provider.exchange_code_for_tokens("bad-code", auth_state)

    def test_parse_token_response_extracts_fields(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """_parse_token_response must extract access_token, refresh_token, and compute expires_at."""
        result = {
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
            "scope": "openid",
        }
        token_info = oauth_provider._parse_token_response(result)
        assert token_info.access_token == "at"
        assert token_info.refresh_token == "rt"
        assert token_info.expires_at > datetime.now(timezone.utc)

    def test_should_refresh_token_true_when_near_expiry(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """A token expiring in < 5 minutes should be flagged for refresh."""
        token_info = TokenInfo(
            access_token="tok",
            refresh_token="rt",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
            scope="openid",
        )
        assert oauth_provider.should_refresh_token(token_info) is True

    def test_should_refresh_token_false_when_fresh(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """A token expiring in > 5 minutes should not be flagged for refresh."""
        token_info = TokenInfo(
            access_token="tok",
            refresh_token="rt",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            scope="openid",
        )
        assert oauth_provider.should_refresh_token(token_info) is False

    def test_should_refresh_token_false_when_no_refresh_token(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """Without a refresh_token, should_refresh_token returns False (can't refresh anyway)."""
        token_info = TokenInfo(
            access_token="tok",
            refresh_token=None,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
            scope="openid",
        )
        assert oauth_provider.should_refresh_token(token_info) is False

    async def test_get_user_profile_no_id_token_raises(
        self, oauth_provider: AzureADOAuthProvider
    ) -> None:
        """get_user_profile with no id_token must raise UserInfoError."""
        with pytest.raises(UserInfoError):
            await oauth_provider.get_user_profile("access-tok", id_token=None)


# =============================================================================
# TokenValidator
# =============================================================================


class TestTokenValidator:
    """Tests for the token validation helper."""

    @pytest.fixture()
    def mock_provider(self) -> MagicMock:
        """A mocked AzureADOAuthProvider for TokenValidator."""
        provider = MagicMock(spec=AzureADOAuthProvider)
        return provider

    @pytest.fixture()
    def validator(self, mock_provider: MagicMock) -> TokenValidator:
        return TokenValidator(mock_provider)

    def test_is_token_valid_true_when_not_expired(self, validator: TokenValidator) -> None:
        """A token with expires_at in the future is valid."""
        token_info = TokenInfo(
            access_token="tok",
            refresh_token="rt",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            scope="openid",
        )
        assert validator.is_token_valid(token_info) is True

    def test_is_token_valid_false_when_expired(self, validator: TokenValidator) -> None:
        """A token with expires_at in the past is invalid."""
        token_info = TokenInfo(
            access_token="tok",
            refresh_token="rt",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            scope="openid",
        )
        assert validator.is_token_valid(token_info) is False

    def test_needs_refresh_delegates_to_provider(
        self, validator: TokenValidator, mock_provider: MagicMock
    ) -> None:
        """needs_refresh must delegate to the provider's should_refresh_token."""
        token_info = TokenInfo(
            access_token="tok",
            refresh_token="rt",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=3),
            scope="openid",
        )
        mock_provider.should_refresh_token.return_value = True
        assert validator.needs_refresh(token_info) is True
        mock_provider.should_refresh_token.assert_called_once_with(token_info)

    async def test_ensure_valid_token_returns_same_when_valid(
        self, validator: TokenValidator, mock_provider: MagicMock
    ) -> None:
        """When the token is still valid and fresh, the same TokenInfo is returned."""
        token_info = TokenInfo(
            access_token="tok",
            refresh_token="rt",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            scope="openid",
        )
        mock_provider.should_refresh_token.return_value = False
        result = await validator.ensure_valid_token(token_info)
        assert result.access_token == "tok"

    async def test_ensure_valid_token_raises_when_expired_no_refresh(
        self, validator: TokenValidator, mock_provider: MagicMock
    ) -> None:
        """An expired token with no refresh_token must raise TokenError."""
        token_info = TokenInfo(
            access_token="tok",
            refresh_token=None,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            scope="openid",
        )
        mock_provider.should_refresh_token.return_value = False
        with pytest.raises(TokenError):
            await validator.ensure_valid_token(token_info)


# =============================================================================
# Exception hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Verify that custom exceptions form the expected inheritance chain."""

    def test_authorization_error_inherits_authentication_error(self) -> None:
        assert issubclass(AuthorizationError, AuthenticationError)

    def test_token_error_inherits_authentication_error(self) -> None:
        assert issubclass(TokenError, AuthenticationError)

    def test_user_info_error_inherits_authentication_error(self) -> None:
        assert issubclass(UserInfoError, AuthenticationError)
