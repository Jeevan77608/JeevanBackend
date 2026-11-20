"""Shared base class for session-scoped repositories."""

from __future__ import annotations

from datetime import timedelta

import redis.asyncio as redis

from src.session.memory_store import InMemoryStore

StoreType = redis.Redis | InMemoryStore


class BaseStore:
    """
    Shared base for all session-scoped repositories.

    Holds a reference to the Redis/InMemoryStore backend and the
    default idle timeout used for TTL on all keys.
    """

    def __init__(self, store: StoreType, idle_timeout: timedelta) -> None:
        self._store = store
        self._idle_timeout = idle_timeout
