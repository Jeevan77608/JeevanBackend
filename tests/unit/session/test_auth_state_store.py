"""Tests for src/session/auth_state_store.py — OAuth state parameter storage."""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.session.auth_state_store import AuthStateStore
from src.session.memory_store import InMemoryStore

STATE_VALUE = "oauth-state-abc123"


@pytest.fixture()
async def store():
    mem = InMemoryStore()
    await mem.initialize()
    auth_store = AuthStateStore(mem, timedelta(hours=1))
    yield auth_store
    await mem.close()


class TestAuthStateStore:
    """Verify AuthStateStore one-time-use OAuth state operations."""

    async def test_save_and_get_auth_state(self, store: AuthStateStore) -> None:
        data = {
            "code_verifier": "verifier-xyz",
            "redirect_uri": "https://example.com/callback",
            "scopes": "openid profile",
        }
        await store.save_auth_state(STATE_VALUE, data)

        result = await store.get_auth_state(STATE_VALUE)
        assert result is not None
        assert result["code_verifier"] == "verifier-xyz"
        assert result["redirect_uri"] == "https://example.com/callback"
        assert result["scopes"] == "openid profile"

    async def test_get_auth_state_consumes_one_time(self, store: AuthStateStore) -> None:
        data = {"code_verifier": "single-use"}
        await store.save_auth_state(STATE_VALUE, data)

        first = await store.get_auth_state(STATE_VALUE)
        assert first is not None
        assert first["code_verifier"] == "single-use"

        second = await store.get_auth_state(STATE_VALUE)
        assert second is None

    async def test_get_auth_state_returns_none_when_missing(self, store: AuthStateStore) -> None:
        result = await store.get_auth_state("never-saved-state")
        assert result is None

    async def test_save_auth_state_with_custom_ttl(self, store: AuthStateStore) -> None:
        data = {"code_verifier": "custom-ttl"}
        await store.save_auth_state(STATE_VALUE, data, ttl_seconds=300)

        result = await store.get_auth_state(STATE_VALUE)
        assert result is not None
        assert result["code_verifier"] == "custom-ttl"

    async def test_save_auth_state_overwrites_previous(self, store: AuthStateStore) -> None:
        await store.save_auth_state(STATE_VALUE, {"version": "old"})
        await store.save_auth_state(STATE_VALUE, {"version": "new"})

        result = await store.get_auth_state(STATE_VALUE)
        assert result is not None
        assert result["version"] == "new"

    async def test_different_states_isolated(self, store: AuthStateStore) -> None:
        await store.save_auth_state("state-a", {"id": "a"})
        await store.save_auth_state("state-b", {"id": "b"})

        result_a = await store.get_auth_state("state-a")
        result_b = await store.get_auth_state("state-b")

        assert result_a is not None
        assert result_b is not None
        assert result_a["id"] == "a"
        assert result_b["id"] == "b"
