"""Conversation history storage."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.session.base_store import BaseStore
from src.session.models import ConversationHistory
from src.session.redis_keys import RedisKeys

if TYPE_CHECKING:
    from src.session.draft_repository import DraftRepository

logger = logging.getLogger(__name__)


class ConversationStore(BaseStore):
    """Repository for conversation history and draft conversation operations."""

    def __init__(
        self,
        store: Any,
        idle_timeout: Any,
        draft_repository: DraftRepository,
    ) -> None:
        super().__init__(store, idle_timeout)
        self._drafts = draft_repository

    async def save_conversation(
        self,
        session_id: str,
        conversation: ConversationHistory,
    ) -> None:
        """Save conversation history for a session."""
        r = self._store
        key = RedisKeys.conversation(session_id)

        conversation.updated_at = datetime.now(timezone.utc)
        conv_data = conversation.model_dump(mode="json")

        await r.setex(
            key,
            self._idle_timeout,
            json.dumps(conv_data, default=str),
        )

    async def get_conversation(self, session_id: str) -> ConversationHistory | None:
        """Retrieve conversation history for a session."""
        r = self._store
        key = RedisKeys.conversation(session_id)

        data = await r.get(key)
        if not data:
            return None

        conv_data = json.loads(data)
        return ConversationHistory.model_validate(conv_data)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationHistory:
        """Add a message to conversation history. Creates conversation if it doesn't exist."""
        conversation = await self.get_conversation(session_id)
        if not conversation:
            conversation = ConversationHistory(session_id=session_id)

        conversation.add_message(role, content, metadata)
        await self.save_conversation(session_id, conversation)
        return conversation

    async def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history (keeps system messages)."""
        conversation = await self.get_conversation(session_id)
        if conversation:
            conversation.clear()
            await self.save_conversation(session_id, conversation)

    # =========================================================================
    # Draft conversation (unsaved chat recovery after refresh)
    # =========================================================================

    async def save_draft_conversation(
        self,
        session_id: str,
        messages_snapshot: list[dict[str, Any]],
        title: str = "",
    ) -> None:
        """Store draft conversation for this session (for restore-after-refresh)."""
        r = self._store
        key = RedisKeys.draft_conversation(session_id)
        payload = {
            "messages_snapshot": messages_snapshot,
            "title": title,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        await r.setex(
            key,
            self._idle_timeout,
            json.dumps(payload, default=str),
        )

    async def get_draft_conversation(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Return draft conversation for this session, or None."""
        r = self._store
        key = RedisKeys.draft_conversation(session_id)
        data = await r.get(key)
        if not data:
            return None
        return json.loads(data)

    async def clear_draft_conversation(self, session_id: str) -> None:
        """Remove draft conversation and clear draft work items."""
        r = self._store
        key = RedisKeys.draft_conversation(session_id)
        await r.delete(key)
        await self._drafts.delete_drafts(session_id)
