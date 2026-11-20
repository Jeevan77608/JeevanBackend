from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from jose import jwt

from src.auth.google_oauth import GoogleOAuthProvider
from src.config.settings import Settings


def test_authorization_url_includes_state_and_pkce(mock_settings: Settings) -> None:
    mock_settings.google_client_id = "google-client-id"
    mock_settings.google_redirect_uri = "http://localhost:3000/api/auth/callback"
    provider = GoogleOAuthProvider(mock_settings)
    auth_state = provider.create_authorization_state()

    url = provider.get_authorization_url(auth_state)
    query = parse_qs(urlparse(url).query)

    assert urlparse(url).hostname == "accounts.google.com"
    assert query["client_id"] == ["google-client-id"]
    assert query["redirect_uri"] == [mock_settings.google_redirect_uri]
    assert query["state"] == [auth_state.state]
    assert query["code_challenge"] == [auth_state.pkce.code_challenge]
    assert query["code_challenge_method"] == ["S256"]
    assert query["scope"] == ["openid email profile"]


@pytest.mark.asyncio
async def test_id_token_verification_checks_at_hash_with_access_token(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = GoogleOAuthProvider(mock_settings)
    provider._certs = {"google-key-id": {"kid": "google-key-id"}}
    monkeypatch.setattr(
        jwt, "get_unverified_header", lambda _: {"kid": "google-key-id", "alg": "RS256"}
    )
    decode = MagicMock(return_value={"iss": "https://accounts.google.com"})
    monkeypatch.setattr(jwt, "decode", decode)

    await provider._verify_id_token(
        AsyncMock(), "google-id-token", "google-access-token"
    )

    assert decode.call_args.kwargs["access_token"] == "google-access-token"
