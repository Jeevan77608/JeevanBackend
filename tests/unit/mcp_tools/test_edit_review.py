"""Tests for src/mcp_tools/edit_review.py — draft edit and review tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.devops import AzureDevOpsError
from src.mcp_tools.base import ToolContext, ToolStatus
from src.mcp_tools.edit_review import (
    DeleteDraftTool,
    GetDraftTool,
    ListDraftsTool,
    PreviewItemsTool,
    UpdateDraftTool,
    ValidateItemsTool,
    create_edit_review_tools,
)
from src.models.work_items import DraftCollection, DraftWorkItem


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def tool_context() -> ToolContext:
    """Fully-populated ToolContext for edit/review tests."""
    return ToolContext(
        session_id="sess-edit-001",
        user_id="user-edit-001",
        organization_url="https://dev.azure.com/TestOrg",
        project_id="TestProject",
        project_name="Test Project",
        access_token="test-access-token",
        conversation_id="conv-edit-001",
    )


@pytest.fixture()
def session_manager() -> AsyncMock:
    """Mock SessionManager with all draft-related methods."""
    mgr = AsyncMock()
    mgr.get_drafts.return_value = None
    mgr.get_draft.return_value = None
    mgr.save_drafts.return_value = None
    mgr.update_draft_item.return_value = None
    mgr.delete_draft.return_value = True
    return mgr


@pytest.fixture()
def sample_draft() -> DraftWorkItem:
    """A single draft for reuse."""
    return DraftWorkItem(
        draft_id="d1",
        work_item_type="User Story",
        title="Test Story",
        description="A test description",
        acceptance_criteria="AC 1",
        priority=2,
        story_points=5.0,
    )


@pytest.fixture()
def sample_collection(sample_draft: DraftWorkItem) -> DraftCollection:
    """DraftCollection containing one sample draft."""
    return DraftCollection(
        session_id="sess-edit-001",
        project_id="TestProject",
        items=[sample_draft],
    )


def _mock_settings() -> MagicMock:
    """Build a MagicMock that mimics get_settings() output."""
    settings = MagicMock()
    settings.auth_enabled = False
    settings.azure_devops = MagicMock()
    return settings


# =============================================================================
# ListDraftsTool
# =============================================================================


class TestListDraftsTool:
    """Tests for ListDraftsTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name, category, and optional work_item_type parameter."""
        tool = ListDraftsTool(session_manager)
        defn = tool.definition

        assert defn.name == "list_drafts"
        assert defn.category == "edit"
        assert len(defn.parameters) == 1
        assert defn.parameters[0].name == "work_item_type"
        assert defn.parameters[0].required is False

    @patch("src.mcp_tools.edit_review.restore_session_drafts_from_conversation", new_callable=AsyncMock)
    async def test_execute_with_no_drafts(
        self,
        mock_restore: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When no drafts exist, returns empty list."""
        session_manager.get_drafts.return_value = None

        tool = ListDraftsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["items"] == []
        assert "Found 0 draft" in result.message
        mock_restore.assert_awaited_once()

    @patch("src.mcp_tools.edit_review.restore_session_drafts_from_conversation", new_callable=AsyncMock)
    async def test_execute_with_drafts(
        self,
        mock_restore: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_collection: DraftCollection,
    ) -> None:
        """When drafts exist, returns all items."""
        session_manager.get_drafts.return_value = sample_collection

        tool = ListDraftsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data["items"]) == 1
        assert result.data["items"][0]["draft_id"] == "d1"
        assert result.data["items"][0]["work_item_type"] == "User Story"
        assert result.data["items"][0]["title"] == "Test Story"
        assert "Found 1 draft" in result.message

    @patch("src.mcp_tools.edit_review.restore_session_drafts_from_conversation", new_callable=AsyncMock)
    async def test_execute_filter_by_work_item_type(
        self,
        mock_restore: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """Filtering by work_item_type returns only matching items."""
        story = DraftWorkItem(
            draft_id="d-story", work_item_type="User Story", title="Story"
        )
        task = DraftWorkItem(
            draft_id="d-task", work_item_type="Task", title="Task"
        )
        collection = DraftCollection(
            session_id="sess-edit-001",
            project_id="TestProject",
            items=[story, task],
        )
        session_manager.get_drafts.return_value = collection

        tool = ListDraftsTool(session_manager)
        result = await tool.execute(tool_context, work_item_type="Task")

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data["items"]) == 1
        assert result.data["items"][0]["draft_id"] == "d-task"

    @patch("src.mcp_tools.edit_review.restore_session_drafts_from_conversation", new_callable=AsyncMock)
    async def test_calls_restore_session_drafts_from_conversation(
        self,
        mock_restore: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """Restore is called before fetching drafts to recover from conversation snapshot."""
        session_manager.get_drafts.return_value = None

        tool = ListDraftsTool(session_manager)
        await tool.execute(tool_context)

        mock_restore.assert_awaited_once_with(tool_context, session_manager)


# =============================================================================
# GetDraftTool
# =============================================================================


class TestGetDraftTool:
    """Tests for GetDraftTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name and required draft_id parameter."""
        tool = GetDraftTool(session_manager)
        defn = tool.definition

        assert defn.name == "get_draft"
        assert defn.category == "edit"
        assert len(defn.parameters) == 1
        assert defn.parameters[0].name == "draft_id"
        assert defn.parameters[0].required is True

    async def test_execute_success(
        self,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_draft: DraftWorkItem,
    ) -> None:
        """When draft exists, returns its full details."""
        session_manager.get_draft.return_value = sample_draft

        tool = GetDraftTool(session_manager)
        result = await tool.execute(tool_context, draft_id="d1")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["draft_id"] == "d1"
        assert result.data["type"] == "User Story"
        assert result.data["title"] == "Test Story"
        assert result.data["priority"] == 2
        assert result.data["story_points"] == 5.0
        assert "Draft: Test Story" in result.message

    async def test_execute_draft_not_found(
        self,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When draft does not exist, returns failure."""
        session_manager.get_draft.return_value = None

        tool = GetDraftTool(session_manager)
        result = await tool.execute(tool_context, draft_id="missing-id")

        assert result.status == ToolStatus.ERROR
        assert "Draft not found" in result.error


# =============================================================================
# UpdateDraftTool
# =============================================================================


class TestUpdateDraftTool:
    """Tests for UpdateDraftTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name and expected parameters."""
        tool = UpdateDraftTool(session_manager)
        defn = tool.definition

        assert defn.name == "update_draft"
        assert defn.category == "edit"
        param_names = [p.name for p in defn.parameters]
        assert "draft_id" in param_names
        assert "title" in param_names
        assert "description" in param_names
        assert "work_item_type" in param_names
        assert "fields" in param_names

        draft_id_param = next(p for p in defn.parameters if p.name == "draft_id")
        assert draft_id_param.required is True

    @patch("src.mcp_tools.edit_review.sync_drafts_to_conversation", new_callable=AsyncMock)
    async def test_execute_updates_title(
        self,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_draft: DraftWorkItem,
    ) -> None:
        """Updating title modifies draft and saves."""
        session_manager.get_draft.return_value = sample_draft

        tool = UpdateDraftTool(session_manager)
        result = await tool.execute(
            tool_context, draft_id="d1", title="Updated Title"
        )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["title"] == "Updated Title"
        session_manager.update_draft_item.assert_awaited_once()
        mock_sync.assert_awaited_once()

    @patch("src.mcp_tools.edit_review.sync_drafts_to_conversation", new_callable=AsyncMock)
    async def test_execute_updates_description(
        self,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_draft: DraftWorkItem,
    ) -> None:
        """Updating description modifies draft and saves."""
        session_manager.get_draft.return_value = sample_draft

        tool = UpdateDraftTool(session_manager)
        result = await tool.execute(
            tool_context, draft_id="d1", description="New desc"
        )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["description"] == "New desc"

    @patch("src.mcp_tools.edit_review.sync_drafts_to_conversation", new_callable=AsyncMock)
    async def test_execute_updates_work_item_type(
        self,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_draft: DraftWorkItem,
    ) -> None:
        """Updating work_item_type modifies draft and saves."""
        session_manager.get_draft.return_value = sample_draft

        tool = UpdateDraftTool(session_manager)
        result = await tool.execute(
            tool_context, draft_id="d1", work_item_type="Epic"
        )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["type"] == "Epic"

    @patch("src.mcp_tools.edit_review.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.edit_review.restore_session_drafts_from_conversation", new_callable=AsyncMock)
    async def test_execute_draft_not_found_tries_restore(
        self,
        mock_restore: AsyncMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_draft: DraftWorkItem,
    ) -> None:
        """When draft not in session, restore from conversation is attempted."""
        # First call returns None, second call after restore returns the draft
        session_manager.get_draft.side_effect = [None, sample_draft]

        tool = UpdateDraftTool(session_manager)
        result = await tool.execute(
            tool_context, draft_id="d1", title="Restored Title"
        )

        assert result.status == ToolStatus.SUCCESS
        mock_restore.assert_awaited_once()
        assert result.data["title"] == "Restored Title"

    @patch("src.mcp_tools.edit_review.restore_session_drafts_from_conversation", new_callable=AsyncMock)
    async def test_execute_draft_still_not_found_after_restore(
        self,
        mock_restore: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When draft not found even after restore attempt, returns failure."""
        session_manager.get_draft.return_value = None

        tool = UpdateDraftTool(session_manager)
        result = await tool.execute(
            tool_context, draft_id="nonexistent", title="X"
        )

        assert result.status == ToolStatus.ERROR
        assert "Draft not found" in result.error


# =============================================================================
# DeleteDraftTool
# =============================================================================


class TestDeleteDraftTool:
    """Tests for DeleteDraftTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name and required draft_id parameter."""
        tool = DeleteDraftTool(session_manager)
        defn = tool.definition

        assert defn.name == "delete_draft"
        assert defn.category == "edit"
        assert len(defn.parameters) == 1
        assert defn.parameters[0].name == "draft_id"
        assert defn.parameters[0].required is True

    @patch("src.mcp_tools.edit_review.sync_drafts_to_conversation", new_callable=AsyncMock)
    async def test_execute_success(
        self,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """Successful delete returns confirmation."""
        session_manager.delete_draft.return_value = True

        tool = DeleteDraftTool(session_manager)
        result = await tool.execute(tool_context, draft_id="d1")

        assert result.status == ToolStatus.SUCCESS
        assert "Deleted draft: d1" in result.message
        session_manager.delete_draft.assert_awaited_once_with(
            session_id="sess-edit-001", draft_id="d1"
        )
        mock_sync.assert_awaited_once()

    async def test_execute_draft_not_found(
        self,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When draft does not exist, returns failure."""
        session_manager.delete_draft.return_value = False

        tool = DeleteDraftTool(session_manager)
        result = await tool.execute(tool_context, draft_id="missing-id")

        assert result.status == ToolStatus.ERROR
        assert "Draft not found" in result.error


# =============================================================================
# PreviewItemsTool
# =============================================================================


class TestPreviewItemsTool:
    """Tests for PreviewItemsTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name and optional draft_ids parameter."""
        tool = PreviewItemsTool(session_manager)
        defn = tool.definition

        assert defn.name == "preview_items"
        assert defn.category == "review"
        assert len(defn.parameters) == 1
        assert defn.parameters[0].name == "draft_ids"
        assert defn.parameters[0].required is False

    async def test_execute_previews_all(
        self,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_collection: DraftCollection,
    ) -> None:
        """Previewing all drafts returns complete list with project info."""
        session_manager.get_drafts.return_value = sample_collection

        tool = PreviewItemsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["count"] == 1
        assert result.data["project"] == "Test Project"
        assert result.data["items"][0]["draft_id"] == "d1"
        assert result.data["items"][0]["target_project"] == "Test Project"
        assert "1 item(s) ready to push" in result.message

    async def test_execute_with_specific_draft_ids_filter(
        self,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """Filtering by draft_ids returns only matching items."""
        d1 = DraftWorkItem(
            draft_id="d1", work_item_type="User Story", title="Story 1"
        )
        d2 = DraftWorkItem(
            draft_id="d2", work_item_type="Task", title="Task 1"
        )
        collection = DraftCollection(
            session_id="sess-edit-001",
            project_id="TestProject",
            items=[d1, d2],
        )
        session_manager.get_drafts.return_value = collection

        tool = PreviewItemsTool(session_manager)
        result = await tool.execute(tool_context, draft_ids=["d2"])

        assert result.status == ToolStatus.SUCCESS
        assert result.data["count"] == 1
        assert result.data["items"][0]["draft_id"] == "d2"

    async def test_execute_no_drafts_returns_failure(
        self,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When no drafts exist, returns failure."""
        session_manager.get_drafts.return_value = None

        tool = PreviewItemsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.ERROR
        assert "No drafts to preview" in result.error


# =============================================================================
# ValidateItemsTool
# =============================================================================


class TestValidateItemsTool:
    """Tests for ValidateItemsTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name and optional draft_ids parameter."""
        tool = ValidateItemsTool(session_manager)
        defn = tool.definition

        assert defn.name == "validate_items"
        assert defn.category == "review"
        assert len(defn.parameters) == 1
        assert defn.parameters[0].name == "draft_ids"
        assert defn.parameters[0].required is False

    @patch("src.mcp_tools.edit_review.get_settings")
    @patch("src.mcp_tools.edit_review.AzureDevOpsClient")
    async def test_execute_all_valid(
        self,
        mock_devops_cls: MagicMock,
        mock_get_settings: MagicMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_collection: DraftCollection,
    ) -> None:
        """When all drafts pass validation, all_valid is True."""
        mock_get_settings.return_value = _mock_settings()

        mock_type = MagicMock()
        mock_type.name = "User Story"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item_types.return_value = [mock_type]
        mock_devops_cls.return_value = mock_client

        session_manager.get_drafts.return_value = sample_collection

        tool = ValidateItemsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["all_valid"] is True
        assert result.data["total"] == 1
        assert result.data["valid_count"] == 1
        assert result.data["results"][0]["valid"] is True
        assert "All items valid" in result.message

    @patch("src.mcp_tools.edit_review.get_settings")
    @patch("src.mcp_tools.edit_review.AzureDevOpsClient")
    async def test_execute_with_invalid_type(
        self,
        mock_devops_cls: MagicMock,
        mock_get_settings: MagicMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """Draft with unrecognised work item type produces validation error."""
        mock_get_settings.return_value = _mock_settings()

        mock_type = MagicMock()
        mock_type.name = "Bug"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item_types.return_value = [mock_type]
        mock_devops_cls.return_value = mock_client

        bad_draft = DraftWorkItem(
            draft_id="d-bad-type",
            work_item_type="NonExistentType",
            title="Valid Title",
        )
        collection = DraftCollection(
            session_id="sess-edit-001",
            project_id="TestProject",
            items=[bad_draft],
        )
        session_manager.get_drafts.return_value = collection

        tool = ValidateItemsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["all_valid"] is False
        errors = result.data["results"][0]["errors"]
        assert any("Invalid work item type" in e for e in errors)

    @patch("src.mcp_tools.edit_review.get_settings")
    @patch("src.mcp_tools.edit_review.AzureDevOpsClient")
    async def test_execute_with_missing_title(
        self,
        mock_devops_cls: MagicMock,
        mock_get_settings: MagicMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """Draft with empty title produces validation error."""
        mock_get_settings.return_value = _mock_settings()

        mock_type = MagicMock()
        mock_type.name = "User Story"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item_types.return_value = [mock_type]
        mock_devops_cls.return_value = mock_client

        empty_title_draft = DraftWorkItem(
            draft_id="d-no-title",
            work_item_type="User Story",
            title="   ",
        )
        collection = DraftCollection(
            session_id="sess-edit-001",
            project_id="TestProject",
            items=[empty_title_draft],
        )
        session_manager.get_drafts.return_value = collection

        tool = ValidateItemsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.data["all_valid"] is False
        errors = result.data["results"][0]["errors"]
        assert any("Title is required" in e for e in errors)

    @patch("src.mcp_tools.edit_review.get_settings")
    @patch("src.mcp_tools.edit_review.AzureDevOpsClient")
    async def test_execute_with_title_too_long(
        self,
        mock_devops_cls: MagicMock,
        mock_get_settings: MagicMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """Draft with title exceeding 255 characters produces validation error."""
        mock_get_settings.return_value = _mock_settings()

        mock_type = MagicMock()
        mock_type.name = "User Story"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item_types.return_value = [mock_type]
        mock_devops_cls.return_value = mock_client

        long_title_draft = DraftWorkItem(
            draft_id="d-long-title",
            work_item_type="User Story",
            title="A" * 300,
        )
        collection = DraftCollection(
            session_id="sess-edit-001",
            project_id="TestProject",
            items=[long_title_draft],
        )
        session_manager.get_drafts.return_value = collection

        tool = ValidateItemsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.data["all_valid"] is False
        errors = result.data["results"][0]["errors"]
        assert any("255 characters" in e for e in errors)

    @patch("src.mcp_tools.edit_review.get_settings")
    @patch("src.mcp_tools.edit_review.AzureDevOpsClient")
    async def test_execute_azure_devops_error_returns_failure(
        self,
        mock_devops_cls: MagicMock,
        mock_get_settings: MagicMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_collection: DraftCollection,
    ) -> None:
        """AzureDevOpsError during validation returns failure."""
        mock_get_settings.return_value = _mock_settings()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item_types.side_effect = AzureDevOpsError(
            "Service unavailable"
        )
        mock_devops_cls.return_value = mock_client

        session_manager.get_drafts.return_value = sample_collection

        tool = ValidateItemsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.ERROR
        assert "Validation failed" in result.error

    async def test_execute_no_drafts_returns_failure(
        self,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When no drafts exist, returns failure."""
        session_manager.get_drafts.return_value = None

        tool = ValidateItemsTool(session_manager)
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.ERROR
        assert "No drafts to validate" in result.error


# =============================================================================
# create_edit_review_tools Factory
# =============================================================================


class TestCreateEditReviewTools:
    """Tests for the create_edit_review_tools factory function."""

    def test_returns_six_tools(self, session_manager: AsyncMock) -> None:
        """Factory returns all six edit/review tools."""
        tools = create_edit_review_tools(session_manager)

        assert len(tools) == 6
        names = {t.definition.name for t in tools}
        assert names == {
            "list_drafts",
            "get_draft",
            "update_draft",
            "delete_draft",
            "preview_items",
            "validate_items",
        }
