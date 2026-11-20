"""Board context storage for AI generation with dynamic types."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.session.base_store import BaseStore
from src.session.redis_keys import RedisKeys

if TYPE_CHECKING:
    from src.models.devops import BoardContext

logger = logging.getLogger(__name__)


class BoardContextStore(BaseStore):
    """Repository for project board context caching."""

    async def store_board_context(
        self,
        session_id: str,
        project_id: str,
        board_context: BoardContext,
    ) -> None:
        """Store BoardContext for a project in the session."""
        r = self._store
        key = RedisKeys.board_context(session_id, project_id)

        context_json = board_context.model_dump_json()

        await r.setex(
            key,
            self._idle_timeout,
            context_json,
        )
        logger.debug(
            "Stored board context for session %s, project %s",
            session_id,
            project_id,
        )

    async def get_board_context(
        self,
        session_id: str,
        project_id: str,
    ) -> BoardContext | None:
        """Retrieve stored BoardContext for a project."""
        from src.models.devops import BoardContext

        r = self._store
        key = RedisKeys.board_context(session_id, project_id)

        context_json = await r.get(key)
        if context_json:
            json_str = (
                context_json.decode("utf-8")
                if isinstance(context_json, bytes)
                else context_json
            )
            return BoardContext.model_validate_json(json_str)
        return None
