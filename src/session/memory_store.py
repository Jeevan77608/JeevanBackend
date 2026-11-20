"""
In-memory session storage for development without Redis.

WARNING: This should only be used for local development!
- Sessions are lost on server restart
- Not suitable for multi-instance deployments
- No persistence guarantees
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StoredItem:
    """Item stored in memory with TTL."""
    data: str
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


class InMemoryStore:
    """
    Thread-safe in-memory key-value store with TTL support.

    Implements the same interface as redis.Redis for session operations.
    """

    def __init__(self) -> None:
        self._store: dict[str, StoredItem] = {}
        self._sets: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Start the background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
        logger.info("In-memory session store initialized (development mode)")

    async def close(self) -> None:
        """Stop cleanup and clear store."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._store.clear()
        self._sets.clear()
        logger.info("In-memory session store closed")

    async def _cleanup_expired(self) -> None:
        """Periodically remove expired items."""
        while True:
            await asyncio.sleep(60)  # Cleanup every minute
            async with self._lock:
                expired = [k for k, v in self._store.items() if v.is_expired()]
                for key in expired:
                    del self._store[key]
                if expired:
                    logger.debug(f"Cleaned up {len(expired)} expired items")

    async def ping(self) -> bool:
        """Check if store is available."""
        return True

    async def get(self, key: str) -> str | None:
        """Get value by key."""
        async with self._lock:
            item = self._store.get(key)
            if item is None or item.is_expired():
                if item is not None:
                    del self._store[key]
                return None
            return item.data

    async def setex(
        self,
        key: str,
        ttl: timedelta | int,
        value: str,
    ) -> None:
        """Set value with TTL."""
        if isinstance(ttl, int):
            ttl = timedelta(seconds=ttl)
        expires_at = datetime.now(timezone.utc) + ttl
        async with self._lock:
            self._store[key] = StoredItem(data=value, expires_at=expires_at)

    async def delete(self, *keys: str) -> int:
        """Delete keys."""
        count = 0
        async with self._lock:
            for key in keys:
                if key in self._store:
                    del self._store[key]
                    count += 1
        return count

    async def expire(self, key: str, ttl: timedelta | int) -> bool:
        """Update TTL for key."""
        if isinstance(ttl, int):
            ttl = timedelta(seconds=ttl)
        async with self._lock:
            item = self._store.get(key)
            if item is None:
                return False
            self._store[key] = StoredItem(
                data=item.data,
                expires_at=datetime.now(timezone.utc) + ttl,
            )
            return True

    # Set operations for user session tracking
    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a set."""
        async with self._lock:
            if key not in self._sets:
                self._sets[key] = set()
            before = len(self._sets[key])
            self._sets[key].update(members)
            return len(self._sets[key]) - before

    async def srem(self, key: str, *members: str) -> int:
        """Remove members from a set."""
        async with self._lock:
            if key not in self._sets:
                return 0
            before = len(self._sets[key])
            self._sets[key] -= set(members)
            return before - len(self._sets[key])

    async def smembers(self, key: str) -> set[str]:
        """Get all members of a set."""
        async with self._lock:
            return set(self._sets.get(key, set()))

    async def keys(self, pattern: str) -> list[str]:
        """Get keys matching a glob-style pattern."""
        import fnmatch

        async with self._lock:
            return [
                key
                for key, item in self._store.items()
                if not item.is_expired() and fnmatch.fnmatch(key, pattern)
            ]

    async def info(self, section: str = "") -> dict[str, Any]:
        """Return store info."""
        return {
            "used_memory_human": f"{len(self._store)} items",
            "connected_clients": 1,
        }

    def pipeline(self, transaction: bool = True) -> InMemoryPipeline:
        """Create a pipeline for batch operations."""
        return InMemoryPipeline(self)


class InMemoryPipeline:
    """Pipeline for batch operations (mimics redis pipeline)."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self._commands: list[tuple[str, tuple]] = []

    def get(self, key: str) -> InMemoryPipeline:
        self._commands.append(("get", (key,)))
        return self

    def delete(self, *keys: str) -> InMemoryPipeline:
        self._commands.append(("delete", keys))
        return self

    async def execute(self) -> list[Any]:
        """Execute all commands and return results."""
        results = []
        for cmd, args in self._commands:
            if cmd == "get":
                results.append(await self._store.get(args[0]))
            elif cmd == "delete":
                results.append(await self._store.delete(*args))
        return results

    async def __aenter__(self) -> InMemoryPipeline:
        return self

    async def __aexit__(self, *args) -> None:
        pass

