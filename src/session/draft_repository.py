"""Draft work items repository."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from src.models.draft_snapshot import DraftSnapshotItem
from src.models.work_items import DraftCollection, DraftWorkItem
from src.session.base_store import BaseStore
from src.session.exceptions import SessionNotFoundError
from src.session.redis_keys import RedisKeys

logger = logging.getLogger(__name__)


class DraftRepository(BaseStore):
    """Repository for draft work item CRUD operations."""

    async def save_drafts(
        self,
        session_id: str,
        drafts: DraftCollection,
    ) -> None:
        """Save draft work items collection for a session."""
        r = self._store
        key = RedisKeys.drafts(session_id)

        drafts.updated_at = datetime.now(timezone.utc)
        draft_data = drafts.model_dump(mode="json")

        await r.setex(
            key,
            self._idle_timeout,
            json.dumps(draft_data, default=str),
        )

    async def get_drafts(self, session_id: str) -> DraftCollection | None:
        """Retrieve draft collection for a session."""
        r = self._store
        key = RedisKeys.drafts(session_id)

        data = await r.get(key)
        if not data:
            return None

        draft_data = json.loads(data)
        return DraftCollection.model_validate(draft_data)

    async def delete_drafts(self, session_id: str) -> None:
        """Delete draft collection for a session."""
        r = self._store
        key = RedisKeys.drafts(session_id)
        await r.delete(key)

    async def replace_drafts_from_snapshot(
        self,
        session_id: str,
        drafts_snapshot: list[dict[str, Any]],
        project_id: str,
    ) -> None:
        """
        Replace session drafts with the given snapshot (e.g. from a loaded conversation).
        So list_drafts returns the same set as the saved conversation.
        """
        items: list[DraftWorkItem] = []
        for raw in drafts_snapshot or []:
            if not isinstance(raw, dict):
                continue
            try:
                snapshot = DraftSnapshotItem.model_validate(raw)
                item = DraftWorkItem.model_validate(snapshot.model_dump())
                items.append(item)
            except ValidationError as e:
                logger.warning(
                    "skip invalid draft item in snapshot draft_id=%s: %s",
                    raw.get("draft_id") or raw.get("id", "?"),
                    e,
                )
        collection = DraftCollection(
            session_id=session_id,
            project_id=project_id,
            items=items,
        )
        await self.save_drafts(session_id, collection)

    async def update_draft_item(
        self,
        session_id: str,
        draft_item: DraftWorkItem,
    ) -> DraftCollection:
        """Update a single draft item in the collection."""
        drafts = await self.get_drafts(session_id)
        if not drafts:
            raise SessionNotFoundError(
                f"No drafts found for session: {session_id[:16]}..."
            )

        for i, item in enumerate(drafts.items):
            if item.draft_id == draft_item.draft_id:
                drafts.items[i] = draft_item
                break
        else:
            drafts.items.append(draft_item)

        await self.save_drafts(session_id, drafts)
        return drafts

    async def get_draft(
        self,
        session_id: str,
        draft_id: str,
    ) -> DraftWorkItem | None:
        """Retrieve a single draft item by ID."""
        drafts = await self.get_drafts(session_id)
        if not drafts:
            return None

        for item in drafts.items:
            if item.draft_id == draft_id:
                return item
        return None

    async def delete_draft(
        self,
        session_id: str,
        draft_id: str,
    ) -> bool:
        """Delete a single draft item by ID."""
        drafts = await self.get_drafts(session_id)
        if not drafts:
            return False

        original_count = len(drafts.items)
        drafts.items = [item for item in drafts.items if item.draft_id != draft_id]

        if len(drafts.items) == original_count:
            return False

        await self.save_drafts(session_id, drafts)
        return True
