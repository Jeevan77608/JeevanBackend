"""Tests for src/session/base_store.py — shared base class for session-scoped repositories."""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.session.base_store import BaseStore
from src.session.memory_store import InMemoryStore


@pytest.fixture()
async def store():
    mem = InMemoryStore()
    await mem.initialize()
    yield mem
    await mem.close()


class TestBaseStore:
    """Verify BaseStore correctly stores the backend reference and idle timeout."""

    async def test_init_stores_reference_and_timeout(self, store: InMemoryStore) -> None:
        timeout = timedelta(hours=1)
        base = BaseStore(store, timeout)
        assert base._store is store
        assert base._idle_timeout == timeout

    async def test_store_property_accessible(self, store: InMemoryStore) -> None:
        timeout = timedelta(minutes=30)
        base = BaseStore(store, timeout)
        assert base._store is store

    async def test_idle_timeout_property_accessible(self, store: InMemoryStore) -> None:
        timeout = timedelta(hours=2)
        base = BaseStore(store, timeout)
        assert base._idle_timeout == timedelta(hours=2)

    async def test_different_timeout_values(self, store: InMemoryStore) -> None:
        for minutes in (5, 30, 60, 120):
            timeout = timedelta(minutes=minutes)
            base = BaseStore(store, timeout)
            assert base._idle_timeout == timeout

    async def test_init_with_zero_timeout(self, store: InMemoryStore) -> None:
        timeout = timedelta(seconds=0)
        base = BaseStore(store, timeout)
        assert base._idle_timeout == timedelta(0)
