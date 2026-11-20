"""
Conversation history MCP tools: save, get, list, and draft (unsaved recovery).

Scoped by context.user_id / context.session_id. Available only when CHAT_SAVE_ENABLED=true.
Draft tools use Redis (session_manager) for restore-after-refresh.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from src.config.settings import get_settings
from src.mcp_tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from src.mcp_tools.registry import register_tools
from src.services.conversation_repository import (
    delete_conversation,
    get_conversation,
    list_conversations,
    save_conversation,
    update_conversation_drafts,
)

if TYPE_CHECKING:
    from src.session.manager import SessionManager

logger = logging.getLogger(__name__)


async def sync_drafts_to_conversation(
    context: ToolContext,
    session_manager: SessionManager,
) -> None:
    """
    Persist current session drafts to the saved conversation in DB immediately.
    Call after any draft mutation (create/update/delete) when context.conversation_id is set.
    So when the user creates 2 epics or deletes 1, the conversation shows the latest state when they return.
    """
    if not _feature_enabled() or not context.conversation_id:
        return
    try:
        drafts = await session_manager.get_drafts(context.session_id)
        snapshot = (
            [item.model_dump(mode="json") for item in drafts.items]
            if drafts
            else []
        )
        update_conversation_drafts(
            context.user_id,
            context.conversation_id,
            snapshot,
        )
    except Exception as e:
        logger.warning("sync_drafts_to_conversation: %s", e)


def _feature_enabled() -> bool:
    return get_settings().chat_save_enabled


async def restore_session_drafts_from_conversation(
    context: ToolContext,
    session_manager: SessionManager,
) -> bool:
    """
    If the session has no drafts but we have a saved conversation with drafts_snapshot,
    restore those drafts into the session. Returns True if a restore was performed.

    Use before get_draft/get_drafts/list_drafts so that when the user has a conversation
    open (e.g. 1 epic + 6 tasks in Cosmos), the session is populated and edit/push see the epic.
    """
    if not _feature_enabled() or not context.conversation_id or not context.user_id or not context.project_id:
        return False
    drafts = await session_manager.get_drafts(context.session_id)
    if drafts and drafts.items:
        return False
    return await _replace_session_drafts_with_conversation(context, session_manager)


async def ensure_session_drafts_from_conversation(
    context: ToolContext,
    session_manager: SessionManager,
) -> bool:
    """
    When a conversation is open, replace session drafts with the conversation's drafts_snapshot
    so push/edit use the same list the UI shows (e.g. 1 epic + 6 tasks). Returns True if replaced.

    Use in push_to_devops so we push what's in the conversation, not a stale session that might
    have only tasks.
    """
    if not _feature_enabled() or not context.conversation_id or not context.user_id or not context.project_id:
        return False
    return await _replace_session_drafts_with_conversation(context, session_manager)


async def _replace_session_drafts_with_conversation(
    context: ToolContext,
    session_manager: SessionManager,
) -> bool:
    """Load conversation's drafts_snapshot and replace session drafts. Returns True if snapshot was applied."""
    doc = get_conversation(context.user_id, context.conversation_id)
    snapshot = getattr(doc, "drafts_snapshot", None) if doc else None
    if not snapshot or not isinstance(snapshot, list) or len(snapshot) == 0:
        return False
    logger.info(
        "restore_session_drafts_from_conversation: restoring %s drafts from conversation %s",
        len(snapshot),
        context.conversation_id[:8],
    )
    await session_manager.replace_drafts_from_snapshot(
        context.session_id,
        snapshot,
        context.project_id,
    )
    return True


class SaveConversationTool(BaseTool):
    """Save or update a chat conversation (markdown + messages snapshot) for the current user."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="save_conversation",
            description=(
                "Save the current chat conversation for the authenticated user. "
                "Stores markdown and a messages snapshot so the user can resume later. "
                "Use when the user explicitly saves or when auto-saving after a turn."
            ),
            parameters=[
                ToolParameter(
                    name="title",
                    type="string",
                    description="Optional display title for the conversation",
                    required=False,
                ),
                ToolParameter(
                    name="markdown",
                    type="string",
                    description="Conversation content as markdown",
                    required=False,
                ),
                ToolParameter(
                    name="messages_snapshot",
                    type="object",
                    description="Array of message objects (role, content, toolCalls, etc.) for UI restore",
                    required=True,
                ),
                ToolParameter(
                    name="conversation_id",
                    type="string",
                    description="If provided, update this conversation; otherwise create new",
                    required=False,
                ),
                ToolParameter(
                    name="context_snapshot",
                    type="object",
                    description="Optional context panel state (organization, project, context_sources, phase, etc.) for restore",
                    required=False,
                ),
                ToolParameter(
                    name="drafts_snapshot",
                    type="object",
                    description="Optional draft work items for Review & Edit panel restore",
                    required=False,
                ),
            ],
            returns="Conversation id and updated_at",
            examples=["Save current chat with title 'Sprint planning'"],
            category="conversation",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        if not _feature_enabled():
            return ToolResult.failure("Chat save is disabled by configuration")
        params = self.validate_params(**kwargs)
        messages_snapshot = params.get("messages_snapshot")
        if isinstance(messages_snapshot, str):
            try:
                messages_snapshot = json.loads(messages_snapshot)
            except json.JSONDecodeError:
                return ToolResult.failure("messages_snapshot must be valid JSON")
        if not isinstance(messages_snapshot, list):
            return ToolResult.failure("messages_snapshot must be an array of message objects")
        title = params.get("title") or ""
        markdown = params.get("markdown") or ""
        conversation_id = params.get("conversation_id")
        context_snapshot = params.get("context_snapshot")
        if isinstance(context_snapshot, str):
            try:
                context_snapshot = json.loads(context_snapshot)
            except json.JSONDecodeError:
                context_snapshot = None
        if not isinstance(context_snapshot, dict):
            context_snapshot = None
        drafts_snapshot = params.get("drafts_snapshot")
        if isinstance(drafts_snapshot, str):
            try:
                drafts_snapshot = json.loads(drafts_snapshot)
            except json.JSONDecodeError:
                drafts_snapshot = None
        if not isinstance(drafts_snapshot, list):
            drafts_snapshot = None
        try:
            doc = save_conversation(
                context.user_id,
                conversation_id=conversation_id,
                session_id=context.session_id,
                title=title,
                markdown=markdown,
                messages_snapshot=messages_snapshot,
                context_snapshot=context_snapshot,
                drafts_snapshot=drafts_snapshot,
            )
            return ToolResult.success(
                data={
                    "conversation_id": doc.id,
                    "title": doc.title,
                    "updated_at": doc.updated_at.isoformat(),
                },
                message="Conversation saved",
            )
        except RuntimeError as e:
            return ToolResult.failure(str(e))
        except Exception as e:
            logger.exception("save_conversation failed")
            return ToolResult.failure(str(e))


class GetConversationTool(BaseTool):
    """Get a saved conversation by id for the current user. When session_manager is set, restores that conversation's drafts into the session so list_drafts matches the saved state (deleted epics stay deleted)."""

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_conversation",
            description=(
                "Load a saved conversation by id for the authenticated user. "
                "Returns markdown and messages_snapshot so the client can restore the chat UI."
            ),
            parameters=[
                ToolParameter(
                    name="conversation_id",
                    type="string",
                    description="Id of the conversation to load",
                    required=True,
                ),
            ],
            returns="Conversation with markdown and messages_snapshot",
            examples=["Load conversation abc-123"],
            category="conversation",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        if not _feature_enabled():
            return ToolResult.failure("Chat save is disabled by configuration")
        params = self.validate_params(**kwargs)
        cid = params["conversation_id"]
        doc = get_conversation(context.user_id, cid)
        if doc is None:
            return ToolResult.failure("Conversation not found or access denied")
        data = {
            "conversation_id": doc.id,
            "title": doc.title,
            "markdown": doc.markdown,
            "messages_snapshot": doc.messages_snapshot,
            "updated_at": doc.updated_at.isoformat(),
        }
        # Support documents created before context_snapshot/drafts_snapshot existed
        context_snapshot = getattr(doc, "context_snapshot", None)
        drafts_snapshot = getattr(doc, "drafts_snapshot", None)
        if context_snapshot is not None:
            data["context_snapshot"] = context_snapshot
        # Always return drafts_snapshot so frontend uses it and never re-derives from messages
        # (avoids deleted/pushed items reappearing when reopening the chat)
        data["drafts_snapshot"] = drafts_snapshot if drafts_snapshot is not None else []
        # Restore this conversation's drafts into the session so list_drafts matches saved state
        if self._session_manager and data["drafts_snapshot"] is not None:
            project_id = context.project_id or context.project_name
            if project_id:
                try:
                    await self._session_manager.replace_drafts_from_snapshot(
                        context.session_id,
                        drafts_snapshot=data["drafts_snapshot"],
                        project_id=project_id,
                    )
                except Exception as e:
                    logger.warning("restore drafts from conversation snapshot: %s", e)
        return ToolResult.success(data=data)


class ListConversationsTool(BaseTool):
    """List saved conversations for the current user, newest first."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_conversations",
            description=(
                "List saved conversations for the authenticated user, newest first. "
                "Used to show the conversation history sidebar."
            ),
            parameters=[
                ToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum number of conversations to return (default 50)",
                    required=False,
                    default=50,
                ),
                ToolParameter(
                    name="days",
                    type="number",
                    description="Only return conversations updated within this many days (default from server config)",
                    required=False,
                ),
            ],
            returns="List of { id, title, updated_at }",
            examples=["List my last 20 conversations"],
            category="conversation",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        if not _feature_enabled():
            return ToolResult.failure("Chat save is disabled by configuration")
        params = self.validate_params(**kwargs)
        limit = int(params.get("limit") or 50)
        limit = min(max(limit, 1), 100)
        days_param = params.get("days")
        days = int(days_param) if days_param is not None else get_settings().chat_history_display_days
        docs = list_conversations(context.user_id, limit=limit, days=days)
        return ToolResult.success(
            data={
                "conversations": [
                    {
                        "id": d.id,
                        "title": d.title or "Untitled",
                        "updated_at": d.updated_at.isoformat(),
                    }
                    for d in docs
                ]
            },
        )


class UpdateConversationTool(BaseTool):
    """Update only context_snapshot and/or drafts_snapshot of an existing conversation; messages are unchanged."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="update_conversation",
            description=(
                "Update only the context panel state and/or draft work items for an existing conversation. "
                "Does not change messages, title, or markdown. Use when drafts are added, edited, or deleted."
            ),
            parameters=[
                ToolParameter(
                    name="conversation_id",
                    type="string",
                    description="Id of the conversation to update",
                    required=True,
                ),
                ToolParameter(
                    name="context_snapshot",
                    type="object",
                    description="Optional context panel state (organization, project, context_sources, phase, etc.)",
                    required=False,
                ),
                ToolParameter(
                    name="drafts_snapshot",
                    type="object",
                    description="Optional draft work items array for Review & Edit panel (e.g. after delete)",
                    required=False,
                ),
            ],
            returns="Success with updated_at",
            examples=["Update drafts after user deletes an epic"],
            category="conversation",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        if not _feature_enabled():
            return ToolResult.failure("Chat save is disabled by configuration")
        params = self.validate_params(**kwargs)
        cid = params["conversation_id"]
        existing = get_conversation(context.user_id, cid)
        if existing is None:
            return ToolResult.failure("Conversation not found or access denied")
        context_snapshot = params.get("context_snapshot")
        if isinstance(context_snapshot, str):
            try:
                context_snapshot = json.loads(context_snapshot)
            except json.JSONDecodeError:
                context_snapshot = None
        if not isinstance(context_snapshot, dict):
            context_snapshot = getattr(existing, "context_snapshot", None)
        # When client sends drafts_snapshot (or draftsSnapshot), always use it (including []) so deletes persist
        drafts_snapshot = params.get("drafts_snapshot") or params.get("draftsSnapshot")
        if "drafts_snapshot" in params or "draftsSnapshot" in params:
            if isinstance(drafts_snapshot, str):
                try:
                    drafts_snapshot = json.loads(drafts_snapshot)
                except json.JSONDecodeError:
                    drafts_snapshot = []
            if not isinstance(drafts_snapshot, list):
                drafts_snapshot = []
        else:
            drafts_snapshot = getattr(existing, "drafts_snapshot", None)
        try:
            doc = save_conversation(
                context.user_id,
                conversation_id=cid,
                session_id=existing.session_id,
                title=existing.title,
                markdown=existing.markdown,
                messages_snapshot=existing.messages_snapshot,
                context_snapshot=context_snapshot,
                drafts_snapshot=drafts_snapshot,
            )
            return ToolResult.success(
                data={
                    "conversation_id": doc.id,
                    "updated_at": doc.updated_at.isoformat(),
                },
                message="Conversation snapshots updated",
            )
        except RuntimeError as e:
            return ToolResult.failure(str(e))
        except Exception as e:
            logger.exception("update_conversation failed")
            return ToolResult.failure(str(e))


class DeleteConversationTool(BaseTool):
    """Delete a saved conversation by id for the current user."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="delete_conversation",
            description=(
                "Delete a saved conversation by id for the authenticated user. "
                "Removes the conversation from history; cannot be undone."
            ),
            parameters=[
                ToolParameter(
                    name="conversation_id",
                    type="string",
                    description="Id of the conversation to delete",
                    required=True,
                ),
            ],
            returns="Success or failure",
            examples=["Delete conversation abc-123"],
            category="conversation",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        if not _feature_enabled():
            return ToolResult.failure("Chat save is disabled by configuration")
        params = self.validate_params(**kwargs)
        cid = params["conversation_id"]
        deleted = delete_conversation(context.user_id, cid)
        if not deleted:
            return ToolResult.failure("Conversation not found or access denied")
        return ToolResult.success(data={"deleted": True}, message="Conversation deleted")


def _conversation_tools_list(session_manager: SessionManager | None = None) -> list[BaseTool]:
    """Build conversation tools; pass session_manager so get_conversation can restore drafts."""
    return [
        SaveConversationTool(),
        GetConversationTool(session_manager),
        ListConversationsTool(),
        UpdateConversationTool(),
        DeleteConversationTool(),
    ]


class SaveDraftConversationTool(BaseTool):
    """Store current conversation as draft for this session (restore after refresh)."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="save_draft_conversation",
            description=(
                "Store the current chat as a draft for this session. "
                "Used so the user can recover the conversation after a refresh if they did not click Save."
            ),
            parameters=[
                ToolParameter(
                    name="messages_snapshot",
                    type="object",
                    description="Array of message objects (role, content, toolCalls, etc.)",
                    required=True,
                ),
                ToolParameter(
                    name="title",
                    type="string",
                    description="Optional display title for the draft",
                    required=False,
                ),
            ],
            returns="Success",
            examples=["Save draft after assistant reply"],
            category="conversation",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        if not _feature_enabled():
            return ToolResult.failure("Chat save is disabled by configuration")
        params = self.validate_params(**kwargs)
        messages_snapshot = params.get("messages_snapshot")
        if isinstance(messages_snapshot, str):
            try:
                messages_snapshot = json.loads(messages_snapshot)
            except json.JSONDecodeError:
                return ToolResult.failure("messages_snapshot must be valid JSON")
        if not isinstance(messages_snapshot, list):
            return ToolResult.failure("messages_snapshot must be an array")
        title = params.get("title") or ""
        try:
            await self._session_manager.save_draft_conversation(
                context.session_id,
                messages_snapshot=messages_snapshot,
                title=title,
            )
            return ToolResult.success(data={"ok": True}, message="Draft saved")
        except Exception as e:
            logger.exception("save_draft_conversation failed")
            return ToolResult.failure(str(e))


class GetDraftConversationTool(BaseTool):
    """Return draft conversation for this session (for restore after refresh)."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_draft_conversation",
            description=(
                "Get the draft (unsaved) conversation for this session, if any. "
                "Used on page load to restore the last conversation after a refresh."
            ),
            parameters=[],
            returns="messages_snapshot and optional title, or empty if no draft",
            examples=["Restore unsaved conversation on load"],
            category="conversation",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        if not _feature_enabled():
            return ToolResult.failure("Chat save is disabled by configuration")
        try:
            draft = await self._session_manager.get_draft_conversation(context.session_id)
            if not draft:
                return ToolResult.success(
                    data={"messages_snapshot": [], "title": ""},
                    message="No draft",
                )
            return ToolResult.success(
                data={
                    "messages_snapshot": draft.get("messages_snapshot") or [],
                    "title": draft.get("title") or "",
                    "updated_at": draft.get("updated_at"),
                },
                message="Draft found",
            )
        except Exception as e:
            logger.exception("get_draft_conversation failed")
            return ToolResult.failure(str(e))


class ClearDraftConversationTool(BaseTool):
    """Clear draft for this session (after New chat or after Save to Cosmos)."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="clear_draft_conversation",
            description=(
                "Clear the draft conversation for this session. "
                "Call after the user starts a new chat or after saving the conversation to history."
            ),
            parameters=[],
            returns="Success",
            examples=["Clear draft after New chat"],
            category="conversation",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        if not _feature_enabled():
            return ToolResult.failure("Chat save is disabled by configuration")
        try:
            await self._session_manager.clear_draft_conversation(context.session_id)
            return ToolResult.success(data={"ok": True}, message="Draft cleared")
        except Exception as e:
            logger.exception("clear_draft_conversation failed")
            return ToolResult.failure(str(e))


@register_tools
def create_conversation_tools(session_manager: SessionManager | None = None) -> list[BaseTool]:
    """Return conversation tools when CHAT_SAVE_ENABLED=true; includes draft tools if session_manager provided."""
    if not _feature_enabled():
        return []
    out: list[BaseTool] = list(_conversation_tools_list(session_manager))
    if session_manager is not None:
        out.extend([
            SaveDraftConversationTool(session_manager),
            GetDraftConversationTool(session_manager),
            ClearDraftConversationTool(session_manager),
        ])
    return out
