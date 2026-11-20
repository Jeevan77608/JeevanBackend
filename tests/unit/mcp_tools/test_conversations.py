"""Tests for src/mcp_tools/conversations.py — conversation history and draft tools."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_tools.base import ToolContext, ToolResult, ToolStatus
from src.mcp_tools.conversations import (
    ClearDraftConversationTool,
    DeleteConversationTool,
    GetConversationTool,
    GetDraftConversationTool,
    ListConversationsTool,
    SaveConversationTool,
    SaveDraftConversationTool,
    UpdateConversationTool,
    _feature_enabled,
    create_conversation_tools,
    ensure_session_drafts_from_conversation,
    restore_session_drafts_from_conversation,
    sync_drafts_to_conversation,
)
from src.models.conversation import ConversationDocument


# =============================================================================
# Helpers
# =============================================================================

_PATCH_SETTINGS = "src.mcp_tools.conversations.get_settings"
_PATCH_SAVE = "src.mcp_tools.conversations.save_conversation"
_PATCH_GET = "src.mcp_tools.conversations.get_conversation"
_PATCH_LIST = "src.mcp_tools.conversations.list_conversations"
_PATCH_DELETE = "src.mcp_tools.conversations.delete_conversation"
_PATCH_UPDATE_DRAFTS = "src.mcp_tools.conversations.update_conversation_drafts"


def _make_context(**overrides: Any) -> ToolContext:
    """Build a ToolContext with all fields populated."""
    defaults = dict(
        session_id="sess-conv-001",
        user_id="user-conv-001",
        organization_url="https://dev.azure.com/TestOrg",
        project_id="TestProject",
        project_name="Test Project",
        access_token="test-access-token",
        conversation_id=None,
    )
    defaults.update(overrides)
    return ToolContext(**defaults)


def _settings_enabled() -> MagicMock:
    """Mock settings with chat_save_enabled=True."""
    s = MagicMock()
    s.chat_save_enabled = True
    s.chat_history_display_days = 30
    return s


def _settings_disabled() -> MagicMock:
    """Mock settings with chat_save_enabled=False."""
    s = MagicMock()
    s.chat_save_enabled = False
    s.chat_history_display_days = 30
    return s


def _make_conversation_doc(
    doc_id: str = "conv-001",
    user_id: str = "user-conv-001",
    title: str = "Test Conversation",
    **overrides: Any,
) -> ConversationDocument:
    """Build a ConversationDocument for testing."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        id=doc_id,
        user_id=user_id,
        session_id="sess-conv-001",
        title=title,
        markdown="# Test\n\nHello",
        messages_snapshot=[{"role": "user", "content": "hi"}],
        context_snapshot={"project": "TestProject"},
        drafts_snapshot=[{"draft_id": "d1", "title": "Draft"}],
        created_at=now,
        updated_at=now,
    )
    defaults.update(overrides)
    return ConversationDocument(**defaults)


# =============================================================================
# _feature_enabled
# =============================================================================


class TestFeatureEnabled:
    """Tests for the _feature_enabled helper."""

    def test_returns_true_when_enabled(self) -> None:
        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            assert _feature_enabled() is True

    def test_returns_false_when_disabled(self) -> None:
        with patch(_PATCH_SETTINGS, return_value=_settings_disabled()):
            assert _feature_enabled() is False


# =============================================================================
# sync_drafts_to_conversation
# =============================================================================


class TestSyncDraftsToConversation:
    """Tests for the sync_drafts_to_conversation helper."""

    async def test_does_nothing_when_feature_disabled(self) -> None:
        """When chat_save_enabled=False, no calls are made."""
        session_manager = AsyncMock()
        ctx = _make_context(conversation_id="conv-001")

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_disabled()),
            patch(_PATCH_UPDATE_DRAFTS) as mock_update,
        ):
            await sync_drafts_to_conversation(ctx, session_manager)

        session_manager.get_drafts.assert_not_awaited()
        mock_update.assert_not_called()

    async def test_does_nothing_when_no_conversation_id(self) -> None:
        """When context has no conversation_id, no calls are made."""
        session_manager = AsyncMock()
        ctx = _make_context(conversation_id=None)

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_UPDATE_DRAFTS) as mock_update,
        ):
            await sync_drafts_to_conversation(ctx, session_manager)

        session_manager.get_drafts.assert_not_awaited()
        mock_update.assert_not_called()

    async def test_calls_update_when_enabled(self) -> None:
        """When enabled and conversation_id is set, update_conversation_drafts is called."""
        session_manager = AsyncMock()
        mock_drafts = MagicMock()
        mock_item = MagicMock()
        mock_item.model_dump.return_value = {"draft_id": "d1", "title": "Test"}
        mock_drafts.items = [mock_item]
        session_manager.get_drafts.return_value = mock_drafts

        ctx = _make_context(conversation_id="conv-001")

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_UPDATE_DRAFTS) as mock_update,
        ):
            await sync_drafts_to_conversation(ctx, session_manager)

        session_manager.get_drafts.assert_awaited_once_with("sess-conv-001")
        mock_update.assert_called_once_with(
            "user-conv-001",
            "conv-001",
            [{"draft_id": "d1", "title": "Test"}],
        )


# =============================================================================
# restore_session_drafts_from_conversation
# =============================================================================


class TestRestoreSessionDraftsFromConversation:
    """Tests for restore_session_drafts_from_conversation."""

    async def test_does_nothing_when_disabled(self) -> None:
        """When feature is disabled, return False immediately."""
        session_manager = AsyncMock()
        ctx = _make_context(conversation_id="conv-001")

        with patch(_PATCH_SETTINGS, return_value=_settings_disabled()):
            result = await restore_session_drafts_from_conversation(ctx, session_manager)

        assert result is False
        session_manager.get_drafts.assert_not_awaited()

    async def test_returns_false_when_session_has_drafts(self) -> None:
        """When session already has drafts, do not restore; return False."""
        session_manager = AsyncMock()
        mock_drafts = MagicMock()
        mock_drafts.items = [MagicMock()]
        session_manager.get_drafts.return_value = mock_drafts

        ctx = _make_context(conversation_id="conv-001")

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_GET) as mock_get_conv,
        ):
            result = await restore_session_drafts_from_conversation(ctx, session_manager)

        assert result is False
        mock_get_conv.assert_not_called()

    async def test_returns_true_when_restoring(self) -> None:
        """When session is empty and conversation has drafts, restore and return True."""
        session_manager = AsyncMock()
        empty_drafts = MagicMock()
        empty_drafts.items = []
        session_manager.get_drafts.return_value = empty_drafts
        session_manager.replace_drafts_from_snapshot.return_value = None

        doc = _make_conversation_doc(drafts_snapshot=[{"draft_id": "d1"}])
        ctx = _make_context(conversation_id="conv-001")

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_GET, return_value=doc),
        ):
            result = await restore_session_drafts_from_conversation(ctx, session_manager)

        assert result is True
        session_manager.replace_drafts_from_snapshot.assert_awaited_once()

    async def test_returns_false_when_no_conversation_id(self) -> None:
        """When conversation_id is None, return False."""
        session_manager = AsyncMock()
        ctx = _make_context(conversation_id=None)

        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            result = await restore_session_drafts_from_conversation(ctx, session_manager)

        assert result is False


# =============================================================================
# ensure_session_drafts_from_conversation
# =============================================================================


class TestEnsureSessionDraftsFromConversation:
    """Tests for ensure_session_drafts_from_conversation."""

    async def test_delegates_to_replace(self) -> None:
        """ensure delegates to _replace_session_drafts_with_conversation."""
        session_manager = AsyncMock()
        session_manager.replace_drafts_from_snapshot.return_value = None

        doc = _make_conversation_doc(drafts_snapshot=[{"draft_id": "d1"}])
        ctx = _make_context(conversation_id="conv-001")

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_GET, return_value=doc),
        ):
            result = await ensure_session_drafts_from_conversation(ctx, session_manager)

        assert result is True
        session_manager.replace_drafts_from_snapshot.assert_awaited_once()

    async def test_returns_false_when_disabled(self) -> None:
        """When feature disabled, return False."""
        session_manager = AsyncMock()
        ctx = _make_context(conversation_id="conv-001")

        with patch(_PATCH_SETTINGS, return_value=_settings_disabled()):
            result = await ensure_session_drafts_from_conversation(ctx, session_manager)

        assert result is False


# =============================================================================
# SaveConversationTool — definition
# =============================================================================


class TestSaveConversationToolDefinition:
    """Tests for SaveConversationTool.definition."""

    def test_name(self) -> None:
        tool = SaveConversationTool()
        assert tool.definition.name == "save_conversation"

    def test_category(self) -> None:
        tool = SaveConversationTool()
        assert tool.definition.category == "conversation"

    def test_messages_snapshot_required(self) -> None:
        tool = SaveConversationTool()
        param = next(p for p in tool.definition.parameters if p.name == "messages_snapshot")
        assert param.required is True


# =============================================================================
# SaveConversationTool — execute
# =============================================================================


class TestSaveConversationToolExecute:
    """Tests for SaveConversationTool.execute."""

    async def test_execute_success(self) -> None:
        """Successfully save a conversation."""
        doc = _make_conversation_doc()

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_SAVE, return_value=doc) as mock_save,
        ):
            tool = SaveConversationTool()
            ctx = _make_context()
            result = await tool.execute(
                ctx,
                messages_snapshot=[{"role": "user", "content": "hi"}],
                title="Test",
            )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["conversation_id"] == "conv-001"
        mock_save.assert_called_once()

    async def test_feature_disabled_returns_failure(self) -> None:
        """When chat_save_enabled=False, return failure."""
        with patch(_PATCH_SETTINGS, return_value=_settings_disabled()):
            tool = SaveConversationTool()
            ctx = _make_context()
            result = await tool.execute(
                ctx,
                messages_snapshot=[{"role": "user", "content": "hi"}],
            )

        assert result.status == ToolStatus.ERROR
        assert "disabled" in result.error.lower()

    async def test_invalid_messages_snapshot_string(self) -> None:
        """When messages_snapshot is not valid JSON string, return failure."""
        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            tool = SaveConversationTool()
            ctx = _make_context()
            result = await tool.execute(ctx, messages_snapshot="not-json{{")

        assert result.status == ToolStatus.ERROR
        assert "valid JSON" in result.error

    async def test_invalid_messages_snapshot_not_list(self) -> None:
        """When messages_snapshot is not a list, return failure."""
        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            tool = SaveConversationTool()
            ctx = _make_context()
            result = await tool.execute(ctx, messages_snapshot={"not": "a list"})

        assert result.status == ToolStatus.ERROR
        assert "array" in result.error.lower()


# =============================================================================
# GetConversationTool — definition
# =============================================================================


class TestGetConversationToolDefinition:
    """Tests for GetConversationTool.definition."""

    def test_name(self) -> None:
        tool = GetConversationTool()
        assert tool.definition.name == "get_conversation"

    def test_category(self) -> None:
        tool = GetConversationTool()
        assert tool.definition.category == "conversation"


# =============================================================================
# GetConversationTool — execute
# =============================================================================


class TestGetConversationToolExecute:
    """Tests for GetConversationTool.execute."""

    async def test_execute_success(self) -> None:
        """Return conversation data when found."""
        doc = _make_conversation_doc()

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_GET, return_value=doc),
        ):
            tool = GetConversationTool()
            ctx = _make_context()
            result = await tool.execute(ctx, conversation_id="conv-001")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["conversation_id"] == "conv-001"
        assert result.data["title"] == "Test Conversation"
        assert result.data["messages_snapshot"] == [{"role": "user", "content": "hi"}]
        assert result.data["context_snapshot"] == {"project": "TestProject"}
        assert result.data["drafts_snapshot"] == [{"draft_id": "d1", "title": "Draft"}]

    async def test_not_found_returns_failure(self) -> None:
        """When conversation does not exist, return failure."""
        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_GET, return_value=None),
        ):
            tool = GetConversationTool()
            ctx = _make_context()
            result = await tool.execute(ctx, conversation_id="nonexistent")

        assert result.status == ToolStatus.ERROR
        assert "not found" in result.error.lower()

    async def test_restores_drafts_when_session_manager_set(self) -> None:
        """When session_manager is provided, drafts from conversation are restored into session."""
        doc = _make_conversation_doc(
            drafts_snapshot=[{"draft_id": "d1", "title": "Restored Draft"}],
        )
        session_manager = AsyncMock()

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_GET, return_value=doc),
        ):
            tool = GetConversationTool(session_manager=session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx, conversation_id="conv-001")

        assert result.status == ToolStatus.SUCCESS
        session_manager.replace_drafts_from_snapshot.assert_awaited_once_with(
            "sess-conv-001",
            drafts_snapshot=[{"draft_id": "d1", "title": "Restored Draft"}],
            project_id="TestProject",
        )


# =============================================================================
# ListConversationsTool — definition
# =============================================================================


class TestListConversationsToolDefinition:
    """Tests for ListConversationsTool.definition."""

    def test_name(self) -> None:
        tool = ListConversationsTool()
        assert tool.definition.name == "list_conversations"

    def test_category(self) -> None:
        tool = ListConversationsTool()
        assert tool.definition.category == "conversation"


# =============================================================================
# ListConversationsTool — execute
# =============================================================================


class TestListConversationsToolExecute:
    """Tests for ListConversationsTool.execute."""

    async def test_execute_success_with_days(self) -> None:
        """Return list of conversations filtered by days."""
        docs = [
            _make_conversation_doc("c1", title="Conv 1"),
            _make_conversation_doc("c2", title="Conv 2"),
        ]

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_LIST, return_value=docs) as mock_list,
        ):
            tool = ListConversationsTool()
            ctx = _make_context()
            result = await tool.execute(ctx, days=7)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data["conversations"]) == 2
        mock_list.assert_called_once_with("user-conv-001", limit=50, days=7)

    async def test_feature_disabled(self) -> None:
        """When disabled, return failure."""
        with patch(_PATCH_SETTINGS, return_value=_settings_disabled()):
            tool = ListConversationsTool()
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.ERROR


# =============================================================================
# UpdateConversationTool — definition
# =============================================================================


class TestUpdateConversationToolDefinition:
    """Tests for UpdateConversationTool.definition."""

    def test_name(self) -> None:
        tool = UpdateConversationTool()
        assert tool.definition.name == "update_conversation"

    def test_category(self) -> None:
        tool = UpdateConversationTool()
        assert tool.definition.category == "conversation"


# =============================================================================
# UpdateConversationTool — execute
# =============================================================================


class TestUpdateConversationToolExecute:
    """Tests for UpdateConversationTool.execute."""

    async def test_execute_success(self) -> None:
        """Update existing conversation snapshots."""
        existing = _make_conversation_doc()
        updated = _make_conversation_doc()

        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_GET, return_value=existing),
            patch(_PATCH_SAVE, return_value=updated),
        ):
            tool = UpdateConversationTool()
            ctx = _make_context()
            result = await tool.execute(
                ctx,
                conversation_id="conv-001",
                drafts_snapshot=[],
            )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["conversation_id"] == "conv-001"

    async def test_conversation_not_found(self) -> None:
        """When conversation does not exist, return failure."""
        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_GET, return_value=None),
        ):
            tool = UpdateConversationTool()
            ctx = _make_context()
            result = await tool.execute(ctx, conversation_id="nonexistent")

        assert result.status == ToolStatus.ERROR
        assert "not found" in result.error.lower()


# =============================================================================
# DeleteConversationTool — definition
# =============================================================================


class TestDeleteConversationToolDefinition:
    """Tests for DeleteConversationTool.definition."""

    def test_name(self) -> None:
        tool = DeleteConversationTool()
        assert tool.definition.name == "delete_conversation"

    def test_category(self) -> None:
        tool = DeleteConversationTool()
        assert tool.definition.category == "conversation"


# =============================================================================
# DeleteConversationTool — execute
# =============================================================================


class TestDeleteConversationToolExecute:
    """Tests for DeleteConversationTool.execute."""

    async def test_execute_success(self) -> None:
        """Successfully delete a conversation."""
        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_DELETE, return_value=True),
        ):
            tool = DeleteConversationTool()
            ctx = _make_context()
            result = await tool.execute(ctx, conversation_id="conv-001")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["deleted"] is True

    async def test_not_found_returns_failure(self) -> None:
        """When conversation does not exist, return failure."""
        with (
            patch(_PATCH_SETTINGS, return_value=_settings_enabled()),
            patch(_PATCH_DELETE, return_value=False),
        ):
            tool = DeleteConversationTool()
            ctx = _make_context()
            result = await tool.execute(ctx, conversation_id="nonexistent")

        assert result.status == ToolStatus.ERROR
        assert "not found" in result.error.lower()

    async def test_feature_disabled(self) -> None:
        """When disabled, return failure."""
        with patch(_PATCH_SETTINGS, return_value=_settings_disabled()):
            tool = DeleteConversationTool()
            ctx = _make_context()
            result = await tool.execute(ctx, conversation_id="conv-001")

        assert result.status == ToolStatus.ERROR


# =============================================================================
# SaveDraftConversationTool — definition and execute
# =============================================================================


class TestSaveDraftConversationTool:
    """Tests for SaveDraftConversationTool."""

    def test_definition_name(self) -> None:
        session_manager = AsyncMock()
        tool = SaveDraftConversationTool(session_manager)
        assert tool.definition.name == "save_draft_conversation"

    def test_definition_category(self) -> None:
        session_manager = AsyncMock()
        tool = SaveDraftConversationTool(session_manager)
        assert tool.definition.category == "conversation"

    async def test_execute_success(self) -> None:
        """Successfully save a draft conversation."""
        session_manager = AsyncMock()
        session_manager.save_draft_conversation.return_value = None

        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            tool = SaveDraftConversationTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(
                ctx,
                messages_snapshot=[{"role": "user", "content": "hello"}],
                title="Draft Title",
            )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["ok"] is True
        session_manager.save_draft_conversation.assert_awaited_once_with(
            "sess-conv-001",
            messages_snapshot=[{"role": "user", "content": "hello"}],
            title="Draft Title",
        )

    async def test_feature_disabled_returns_failure(self) -> None:
        """When disabled, return failure."""
        session_manager = AsyncMock()

        with patch(_PATCH_SETTINGS, return_value=_settings_disabled()):
            tool = SaveDraftConversationTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(
                ctx,
                messages_snapshot=[{"role": "user", "content": "hi"}],
            )

        assert result.status == ToolStatus.ERROR


# =============================================================================
# GetDraftConversationTool — definition and execute
# =============================================================================


class TestGetDraftConversationTool:
    """Tests for GetDraftConversationTool."""

    def test_definition_name(self) -> None:
        session_manager = AsyncMock()
        tool = GetDraftConversationTool(session_manager)
        assert tool.definition.name == "get_draft_conversation"

    def test_definition_category(self) -> None:
        session_manager = AsyncMock()
        tool = GetDraftConversationTool(session_manager)
        assert tool.definition.category == "conversation"

    async def test_execute_success_with_draft(self) -> None:
        """Return draft conversation data when it exists."""
        session_manager = AsyncMock()
        session_manager.get_draft_conversation.return_value = {
            "messages_snapshot": [{"role": "user", "content": "saved"}],
            "title": "My Draft",
            "updated_at": "2026-01-01T00:00:00Z",
        }

        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            tool = GetDraftConversationTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["title"] == "My Draft"
        assert len(result.data["messages_snapshot"]) == 1

    async def test_execute_success_no_draft(self) -> None:
        """Return empty when no draft exists."""
        session_manager = AsyncMock()
        session_manager.get_draft_conversation.return_value = None

        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            tool = GetDraftConversationTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["messages_snapshot"] == []
        assert result.message == "No draft"


# =============================================================================
# ClearDraftConversationTool — definition and execute
# =============================================================================


class TestClearDraftConversationTool:
    """Tests for ClearDraftConversationTool."""

    def test_definition_name(self) -> None:
        session_manager = AsyncMock()
        tool = ClearDraftConversationTool(session_manager)
        assert tool.definition.name == "clear_draft_conversation"

    def test_definition_category(self) -> None:
        session_manager = AsyncMock()
        tool = ClearDraftConversationTool(session_manager)
        assert tool.definition.category == "conversation"

    async def test_execute_success(self) -> None:
        """Successfully clear a draft conversation."""
        session_manager = AsyncMock()
        session_manager.clear_draft_conversation.return_value = None

        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            tool = ClearDraftConversationTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["ok"] is True
        session_manager.clear_draft_conversation.assert_awaited_once_with("sess-conv-001")

    async def test_feature_disabled(self) -> None:
        """When disabled, return failure."""
        session_manager = AsyncMock()

        with patch(_PATCH_SETTINGS, return_value=_settings_disabled()):
            tool = ClearDraftConversationTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.ERROR


# =============================================================================
# create_conversation_tools factory
# =============================================================================


class TestCreateConversationTools:
    """Tests for the create_conversation_tools factory function."""

    def test_returns_empty_when_feature_disabled(self) -> None:
        """When chat_save_enabled=False, factory returns empty list."""
        with patch(_PATCH_SETTINGS, return_value=_settings_disabled()):
            tools = create_conversation_tools(session_manager=AsyncMock())
        assert tools == []

    def test_returns_tools_when_enabled_with_session_manager(self) -> None:
        """When enabled with session_manager, returns all tools including draft tools."""
        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            tools = create_conversation_tools(session_manager=AsyncMock())

        tool_names = {t.definition.name for t in tools}
        # Core conversation tools
        assert "save_conversation" in tool_names
        assert "get_conversation" in tool_names
        assert "list_conversations" in tool_names
        assert "update_conversation" in tool_names
        assert "delete_conversation" in tool_names
        # Draft tools (require session_manager)
        assert "save_draft_conversation" in tool_names
        assert "get_draft_conversation" in tool_names
        assert "clear_draft_conversation" in tool_names

    def test_returns_tools_without_draft_when_no_session_manager(self) -> None:
        """When enabled but no session_manager, draft tools are excluded."""
        with patch(_PATCH_SETTINGS, return_value=_settings_enabled()):
            tools = create_conversation_tools(session_manager=None)

        tool_names = {t.definition.name for t in tools}
        assert "save_conversation" in tool_names
        assert "get_conversation" in tool_names
        # Draft tools should NOT be present
        assert "save_draft_conversation" not in tool_names
        assert "get_draft_conversation" not in tool_names
        assert "clear_draft_conversation" not in tool_names
