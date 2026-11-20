"""Tests for src/models/user.py — UserProfile, TokenInfo, AuthState, SessionContext."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.user import AuthState, SessionContext, TokenInfo, UserProfile


# =============================================================================
# UserProfile
# =============================================================================


class TestUserProfile:
    """Tests for the UserProfile model."""

    def test_creation_with_required_fields(self) -> None:
        user = UserProfile(
            id="u-1",
            display_name="Alice",
            email="alice@example.com",
            user_principal_name="alice@example.com",
            tenant_id="tid-001",
        )
        assert user.id == "u-1"
        assert user.display_name == "Alice"
        assert user.email == "alice@example.com"
        assert user.user_principal_name == "alice@example.com"
        assert user.tenant_id == "tid-001"

    def test_default_groups_empty_list(self) -> None:
        user = UserProfile(
            id="u-1",
            display_name="Alice",
            email="a@b.com",
            user_principal_name="a@b.com",
            tenant_id="tid",
        )
        assert user.groups == []

    def test_creation_with_groups(self) -> None:
        user = UserProfile(
            id="u-1",
            display_name="Bob",
            email="bob@co.com",
            user_principal_name="bob@co.com",
            tenant_id="tid",
            groups=["admins", "devs"],
        )
        assert user.groups == ["admins", "devs"]

    def test_serialization_model_dump(self) -> None:
        user = UserProfile(
            id="u-2",
            display_name="Carol",
            email="carol@co.com",
            user_principal_name="carol@co.com",
            tenant_id="tid-999",
            groups=["qa"],
        )
        data = user.model_dump()
        assert data["id"] == "u-2"
        assert data["display_name"] == "Carol"
        assert data["groups"] == ["qa"]
        assert isinstance(data, dict)

    def test_serialization_round_trip(self) -> None:
        user = UserProfile(
            id="u-3",
            display_name="Dave",
            email="d@e.com",
            user_principal_name="d@e.com",
            tenant_id="t",
        )
        reconstructed = UserProfile(**user.model_dump())
        assert reconstructed == user


# =============================================================================
# TokenInfo
# =============================================================================


class TestTokenInfo:
    """Tests for the TokenInfo model."""

    def test_is_expired_with_past_date(self) -> None:
        token = TokenInfo(
            access_token="tok",
            expires_at=datetime.utcnow() - timedelta(hours=1),
            scope="openid",
        )
        assert token.is_expired is True

    def test_is_not_expired_with_future_date(self) -> None:
        token = TokenInfo(
            access_token="tok",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            scope="openid",
        )
        assert token.is_expired is False

    def test_default_token_type_is_bearer(self) -> None:
        token = TokenInfo(
            access_token="tok",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            scope="openid",
        )
        assert token.token_type == "Bearer"

    def test_custom_token_type(self) -> None:
        token = TokenInfo(
            access_token="tok",
            token_type="MAC",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            scope="openid",
        )
        assert token.token_type == "MAC"

    def test_optional_fields_default_none(self) -> None:
        token = TokenInfo(
            access_token="tok",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            scope="openid",
        )
        assert token.refresh_token is None
        assert token.id_token is None

    def test_refresh_and_id_token_set(self) -> None:
        token = TokenInfo(
            access_token="a",
            refresh_token="r",
            id_token="i",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            scope="s",
        )
        assert token.refresh_token == "r"
        assert token.id_token == "i"


# =============================================================================
# AuthState
# =============================================================================


class TestAuthState:
    """Tests for the AuthState model."""

    def test_creation(self) -> None:
        auth = AuthState(
            state="random-state",
            code_verifier="pkce-verifier",
            redirect_uri="http://localhost/callback",
        )
        assert auth.state == "random-state"
        assert auth.code_verifier == "pkce-verifier"
        assert auth.redirect_uri == "http://localhost/callback"

    def test_default_created_at_is_set(self) -> None:
        before = datetime.utcnow()
        auth = AuthState(
            state="s",
            code_verifier="v",
            redirect_uri="http://example.com",
        )
        after = datetime.utcnow()
        assert before <= auth.created_at <= after

    def test_explicit_created_at(self) -> None:
        fixed = datetime(2025, 6, 15, 12, 0, 0)
        auth = AuthState(
            state="s",
            code_verifier="v",
            redirect_uri="http://example.com",
            created_at=fixed,
        )
        assert auth.created_at == fixed


# =============================================================================
# SessionContext
# =============================================================================


class TestSessionContext:
    """Tests for the SessionContext model."""

    def test_creation_with_all_fields(self, sample_user: UserProfile, sample_token_info: TokenInfo) -> None:
        ctx = SessionContext(
            session_id="sess-1",
            user=sample_user,
            token_info=sample_token_info,
            selected_org="https://dev.azure.com/Org",
            selected_project="Proj",
        )
        assert ctx.session_id == "sess-1"
        assert ctx.user.id == sample_user.id
        assert ctx.selected_org == "https://dev.azure.com/Org"
        assert ctx.selected_project == "Proj"

    def test_default_optional_fields_are_none(self, sample_user: UserProfile, sample_token_info: TokenInfo) -> None:
        ctx = SessionContext(
            session_id="sess-2",
            user=sample_user,
            token_info=sample_token_info,
        )
        assert ctx.selected_org is None
        assert ctx.selected_project is None

    def test_default_metadata_is_empty_dict(self, sample_user: UserProfile, sample_token_info: TokenInfo) -> None:
        ctx = SessionContext(
            session_id="sess-3",
            user=sample_user,
            token_info=sample_token_info,
        )
        assert ctx.metadata == {}

    def test_metadata_accepts_arbitrary_keys(self, sample_user: UserProfile, sample_token_info: TokenInfo) -> None:
        ctx = SessionContext(
            session_id="sess-4",
            user=sample_user,
            token_info=sample_token_info,
            metadata={"theme": "dark", "retries": 3},
        )
        assert ctx.metadata["theme"] == "dark"
        assert ctx.metadata["retries"] == 3

    def test_default_timestamps_set_automatically(self, sample_user: UserProfile, sample_token_info: TokenInfo) -> None:
        before = datetime.utcnow()
        ctx = SessionContext(
            session_id="sess-5",
            user=sample_user,
            token_info=sample_token_info,
        )
        after = datetime.utcnow()
        assert before <= ctx.created_at <= after
        assert before <= ctx.last_accessed <= after
