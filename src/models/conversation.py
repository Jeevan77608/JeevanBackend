"""
Conversation document model for Cosmos DB persistence.

Stored with partition key user_id. TTL set from CHAT_RETENTION_DAYS.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ConversationDocument(BaseModel):
    """
    One conversation saved in Cosmos DB.

    - id: unique conversation id (also Cosmos document id)
    - user_id: partition key; from session/JWT only
    - session_id: optional session when created
    - title: optional display title
    - markdown: full conversation as markdown
    - messages_snapshot: list of message dicts for UI restore
    - created_at, updated_at: timestamps
    - ttl: seconds until Cosmos auto-deletes (optional)
    """

    id: str = Field(..., description="Conversation/document ID")
    user_id: str = Field(..., description="Partition key; from auth only")
    session_id: str | None = Field(default=None, description="Session when saved")
    title: str = Field(default="", description="Display title")
    markdown: str = Field(default="", description="Conversation as markdown")
    messages_snapshot: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Messages for UI restore (role, content, toolCalls, etc.)",
    )
    context_snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Context panel state (organization, project, etc.) for restore",
    )
    drafts_snapshot: list[dict[str, Any]] | None = Field(
        default=None,
        description="Draft work items for Review & Edit panel; used so deletes persist across reloads",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Created timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last updated timestamp",
    )
    ttl: int | None = Field(
        default=None,
        description="Cosmos TTL in seconds; document deleted after created_at + ttl",
    )

    def to_cosmos_item(self) -> dict[str, Any]:
        """Convert to dict for Cosmos (with _ts etc. handled by SDK)."""
        d = self.model_dump(mode="json")
        if self.ttl is not None:
            d["ttl"] = self.ttl
        return d

    @classmethod
    def from_cosmos_item(cls, item: dict[str, Any]) -> ConversationDocument:
        """Build from Cosmos document (may include _ts, _etag)."""
        # Exclude Cosmos system fields
        allowed = {
            "id", "user_id", "session_id", "title", "markdown",
            "messages_snapshot", "context_snapshot", "drafts_snapshot",
            "created_at", "updated_at", "ttl",
        }
        data = {k: v for k, v in item.items() if k in allowed and not k.startswith("_")}
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            )
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(
                data["updated_at"].replace("Z", "+00:00")
            )
        return cls.model_validate(data)
