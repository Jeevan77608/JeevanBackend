"""OAuth auth state storage for CSRF protection."""

from __future__ import annotations

import json
from typing import Any

from src.session.base_store import BaseStore
from src.session.redis_keys import RedisKeys


class AuthStateStore(BaseStore):
    """Repository for OAuth state parameter storage (one-time use)."""

    async def save_auth_state(
        self,
        state: str,
        data: dict[str, Any],
        ttl_seconds: int = 600,  # 10 minutes
    ) -> None:
        """Save OAuth state for CSRF protection."""
        r = self._store
        key = RedisKeys.auth_state(state)

        await r.setex(
            key,
            ttl_seconds,
            json.dumps(data, default=str),
        )

    async def get_auth_state(self, state: str) -> dict[str, Any] | None:
        """Retrieve and consume OAuth state (one-time use)."""
        r = self._store
        key = RedisKeys.auth_state(state)

        # Get and delete atomically using pipeline
        async with r.pipeline(transaction=True) as pipe:
            pipe.get(key)
            pipe.delete(key)
            results = await pipe.execute()

        data = results[0]
        if not data:
            return None

        return json.loads(data)
