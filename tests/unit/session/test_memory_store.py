"""Tests for src/session/memory_store.py — in-memory key-value store with TTL."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.session.memory_store import InMemoryPipeline, InMemoryStore, StoredItem


def _fresh_store() -> InMemoryStore:
    """Create a bare InMemoryStore without starting the cleanup task."""
    return InMemoryStore()


# =============================================================================
# InMemoryStore.ping
# =============================================================================


class TestPing:
    """Tests for the ping health-check method."""

    async def test_ping_returns_true(self) -> None:
        """ping always returns True for the in-memory store."""
        store = _fresh_store()
        assert await store.ping() is True


# =============================================================================
# setex / get — basic roundtrip
# =============================================================================


class TestSetexGet:
    """Tests for basic set-with-expiry and get operations."""

    async def test_basic_roundtrip(self) -> None:
        """setex stores a value that get retrieves."""
        store = _fresh_store()
        await store.setex("key1", 300, "value1")
        assert await store.get("key1") == "value1"

    async def test_ttl_as_int(self) -> None:
        """An integer TTL (seconds) is accepted and the value is accessible."""
        store = _fresh_store()
        await store.setex("k", 60, "v")
        assert await store.get("k") == "v"

    async def test_ttl_as_timedelta(self) -> None:
        """A timedelta TTL is accepted and the value is accessible."""
        store = _fresh_store()
        await store.setex("k", timedelta(minutes=5), "v")
        assert await store.get("k") == "v"

    async def test_get_missing_key_returns_none(self) -> None:
        """Getting a key that was never set returns None."""
        store = _fresh_store()
        assert await store.get("nonexistent") is None

    async def test_overwrite_existing_key(self) -> None:
        """Setting the same key again overwrites the previous value."""
        store = _fresh_store()
        await store.setex("k", 300, "old")
        await store.setex("k", 300, "new")
        assert await store.get("k") == "new"


# =============================================================================
# get — TTL expiry
# =============================================================================


class TestGetExpiry:
    """Tests for automatic expiry on get."""

    async def test_expired_item_returns_none(self) -> None:
        """An item set with a 1-second TTL returns None after sleeping 1.1 seconds."""
        store = _fresh_store()
        await store.setex("ephemeral", 1, "gone-soon")
        await asyncio.sleep(1.1)
        assert await store.get("ephemeral") is None


# =============================================================================
# delete
# =============================================================================


class TestDelete:
    """Tests for the delete operation."""

    async def test_delete_single_key(self) -> None:
        """Deleting a single existing key returns 1."""
        store = _fresh_store()
        await store.setex("to-delete", 300, "val")
        count = await store.delete("to-delete")
        assert count == 1
        assert await store.get("to-delete") is None

    async def test_delete_multiple_keys(self) -> None:
        """Deleting multiple existing keys returns the total count."""
        store = _fresh_store()
        await store.setex("a", 300, "1")
        await store.setex("b", 300, "2")
        await store.setex("c", 300, "3")
        count = await store.delete("a", "b", "c")
        assert count == 3

    async def test_delete_nonexistent_key_returns_zero(self) -> None:
        """Deleting a key that does not exist returns 0."""
        store = _fresh_store()
        count = await store.delete("ghost")
        assert count == 0

    async def test_delete_mix_of_existing_and_missing(self) -> None:
        """Deleting a mix of existing and missing keys counts only the existing ones."""
        store = _fresh_store()
        await store.setex("exists", 300, "yes")
        count = await store.delete("exists", "nope")
        assert count == 1


# =============================================================================
# expire
# =============================================================================


class TestExpire:
    """Tests for updating TTL on an existing key."""

    async def test_expire_updates_ttl(self) -> None:
        """expire returns True and the key remains accessible after TTL update."""
        store = _fresh_store()
        await store.setex("k", 1, "v")
        result = await store.expire("k", 300)
        assert result is True
        # Key should still be alive after the original 1-second TTL
        await asyncio.sleep(1.1)
        assert await store.get("k") == "v"

    async def test_expire_missing_key_returns_false(self) -> None:
        """expire on a non-existent key returns False."""
        store = _fresh_store()
        result = await store.expire("missing", 300)
        assert result is False


# =============================================================================
# sadd / smembers
# =============================================================================


class TestSaddSmembers:
    """Tests for set-add and set-members operations."""

    async def test_sadd_returns_new_member_count(self) -> None:
        """sadd returns the number of newly added members."""
        store = _fresh_store()
        added = await store.sadd("myset", "a", "b", "c")
        assert added == 3

    async def test_sadd_duplicate_not_counted(self) -> None:
        """Adding an already-present member returns 0 for that member."""
        store = _fresh_store()
        await store.sadd("myset", "a")
        added = await store.sadd("myset", "a")
        assert added == 0

    async def test_smembers_returns_all_members(self) -> None:
        """smembers returns the full set of added members."""
        store = _fresh_store()
        await store.sadd("myset", "x", "y")
        members = await store.smembers("myset")
        assert members == {"x", "y"}


# =============================================================================
# srem
# =============================================================================


class TestSrem:
    """Tests for set-remove operations."""

    async def test_srem_removes_member(self) -> None:
        """srem returns count of removed members and the set shrinks."""
        store = _fresh_store()
        await store.sadd("s", "a", "b")
        removed = await store.srem("s", "a")
        assert removed == 1
        assert await store.smembers("s") == {"b"}

    async def test_srem_nonexistent_member_returns_zero(self) -> None:
        """Removing a member that is not in the set returns 0."""
        store = _fresh_store()
        await store.sadd("s", "a")
        removed = await store.srem("s", "z")
        assert removed == 0


# =============================================================================
# smembers — empty set
# =============================================================================


class TestSmembersEmpty:
    """Tests for smembers on missing / empty sets."""

    async def test_smembers_missing_set_returns_empty(self) -> None:
        """smembers for a key that was never created returns an empty set."""
        store = _fresh_store()
        result = await store.smembers("no-such-set")
        assert result == set()


# =============================================================================
# keys — pattern matching
# =============================================================================


class TestKeys:
    """Tests for glob-style key pattern matching."""

    async def test_keys_glob_pattern(self) -> None:
        """keys('test:*') matches only keys with the 'test:' prefix."""
        store = _fresh_store()
        await store.setex("test:1", 300, "a")
        await store.setex("test:2", 300, "b")
        await store.setex("other:1", 300, "c")
        matched = await store.keys("test:*")
        assert sorted(matched) == ["test:1", "test:2"]

    async def test_keys_no_match_returns_empty(self) -> None:
        """A pattern that matches nothing returns an empty list."""
        store = _fresh_store()
        await store.setex("foo", 300, "bar")
        matched = await store.keys("zzz:*")
        assert matched == []


# =============================================================================
# info
# =============================================================================


class TestInfo:
    """Tests for the info introspection method."""

    async def test_info_returns_dict_with_expected_keys(self) -> None:
        """info returns a dict with 'used_memory_human' and 'connected_clients'."""
        store = _fresh_store()
        result = await store.info()
        assert isinstance(result, dict)
        assert "used_memory_human" in result
        assert "connected_clients" in result


# =============================================================================
# StoredItem.is_expired
# =============================================================================


class TestStoredItemIsExpired:
    """Tests for the StoredItem.is_expired method."""

    def test_expired_item_returns_true(self) -> None:
        """An item whose expires_at is in the past reports as expired."""
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        item = StoredItem(data="old", expires_at=past)
        assert item.is_expired() is True

    def test_fresh_item_returns_false(self) -> None:
        """An item whose expires_at is in the future reports as not expired."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        item = StoredItem(data="fresh", expires_at=future)
        assert item.is_expired() is False


# =============================================================================
# InMemoryPipeline
# =============================================================================


class TestInMemoryPipeline:
    """Tests for the batch-operation pipeline."""

    async def test_pipeline_batch_get_and_delete(self) -> None:
        """Pipeline executes queued get and delete commands in order."""
        store = _fresh_store()
        await store.setex("pk1", 300, "pv1")
        await store.setex("pk2", 300, "pv2")

        pipe = store.pipeline()
        pipe.get("pk1")
        pipe.get("pk2")
        pipe.delete("pk1")
        results = await pipe.execute()

        assert results == ["pv1", "pv2", 1]
        # pk1 should now be deleted
        assert await store.get("pk1") is None

    async def test_pipeline_context_manager(self) -> None:
        """Pipeline can be used as an async context manager."""
        store = _fresh_store()
        await store.setex("cm_key", 300, "cm_val")

        async with store.pipeline() as pipe:
            pipe.get("cm_key")
            results = await pipe.execute()

        assert results == ["cm_val"]

    async def test_pipeline_get_missing_key(self) -> None:
        """Pipeline get for a missing key returns None in the results list."""
        store = _fresh_store()
        pipe = store.pipeline()
        pipe.get("does-not-exist")
        results = await pipe.execute()
        assert results == [None]

    async def test_pipeline_returns_inmemory_pipeline(self) -> None:
        """store.pipeline() returns an InMemoryPipeline instance."""
        store = _fresh_store()
        pipe = store.pipeline()
        assert isinstance(pipe, InMemoryPipeline)
