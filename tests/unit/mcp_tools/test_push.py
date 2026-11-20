"""Tests for src/mcp_tools/push.py — PushToDevOpsTool, PushSingleTool, _resolve_work_item_type."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_tools.base import ToolContext, ToolResult, ToolStatus
from src.mcp_tools.push import (
    PushSingleTool,
    PushToDevOpsTool,
    _WIT_TYPE_ALTERNATIVES,
    _resolve_work_item_type,
)
from src.models.work_items import DraftCollection, DraftStatus, DraftWorkItem


# =============================================================================
# Helpers
# =============================================================================


def _make_context(**overrides: Any) -> ToolContext:
    """Build a ToolContext with all fields populated; override any via kwargs."""
    defaults = dict(
        session_id="sess-push-001",
        user_id="user-push-001",
        organization_url="https://dev.azure.com/TestOrg",
        project_id="TestProject",
        project_name="Test Project",
        access_token="test-access-token",
        conversation_id=None,
    )
    defaults.update(overrides)
    return ToolContext(**defaults)


def _make_draft(**overrides: Any) -> DraftWorkItem:
    """Build a DraftWorkItem with sensible defaults."""
    defaults = dict(
        draft_id="draft-001",
        work_item_type="User Story",
        title="Draft Title",
        description="Draft description",
        priority=2,
        story_points=5.0,
        tags=["mvp"],
        status=DraftStatus.DRAFT,
    )
    defaults.update(overrides)
    return DraftWorkItem(**defaults)


def _make_draft_collection(
    drafts: list[DraftWorkItem] | None = None,
    session_id: str = "sess-push-001",
) -> DraftCollection:
    """Build a DraftCollection wrapping given drafts."""
    return DraftCollection(
        session_id=session_id,
        project_id="TestProject",
        items=drafts or [],
    )


def _mock_work_item_types() -> list[MagicMock]:
    """Return a list of mock WorkItemType objects with .name attributes."""
    names = ["User Story", "Task", "Bug", "Epic"]
    types = []
    for n in names:
        t = MagicMock()
        t.name = n
        types.append(t)
    return types


def _mock_create_work_item_response(
    work_item_id: int = 42,
    title: str = "Draft Title",
    work_item_type: str = "User Story",
) -> dict[str, Any]:
    """Return a dict matching Azure DevOps create_work_item API response."""
    return {
        "id": work_item_id,
        "url": f"https://dev.azure.com/TestOrg/_workitems/edit/{work_item_id}",
        "fields": {
            "System.Title": title,
            "System.WorkItemType": work_item_type,
        },
    }


def _build_patchers() -> dict[str, Any]:
    """Build the standard patchers for push tool tests."""
    return {
        "client_cls": "src.mcp_tools.push.AzureDevOpsClient",
        "get_settings": "src.mcp_tools.push.get_settings",
        "ensure": "src.mcp_tools.push.ensure_session_drafts_from_conversation",
        "restore": "src.mcp_tools.push.restore_session_drafts_from_conversation",
        "sync": "src.mcp_tools.push.sync_drafts_to_conversation",
    }


def _default_settings_mock() -> MagicMock:
    """Return a mock settings object with push-relevant fields."""
    settings = MagicMock()
    settings.default_push_tag = "@AIGenerated"
    settings.azure_devops = MagicMock()
    settings.auth_enabled = False
    return settings


# =============================================================================
# _WIT_TYPE_ALTERNATIVES dict
# =============================================================================


class TestWitTypeAlternatives:
    """Tests for the _WIT_TYPE_ALTERNATIVES mapping."""

    def test_known_keys_present(self) -> None:
        """All expected draft types have entries in the alternatives dict."""
        expected_keys = {"User Story", "Issue", "Task", "Epic", "Feature", "Bug"}
        assert expected_keys == set(_WIT_TYPE_ALTERNATIVES.keys())

    def test_user_story_alternatives(self) -> None:
        assert _WIT_TYPE_ALTERNATIVES["User Story"] == [
            "User Story", "Issue", "Product Backlog Item", "Requirement",
        ]

    def test_epic_alternatives(self) -> None:
        assert _WIT_TYPE_ALTERNATIVES["Epic"] == ["Epic", "Feature", "Issue"]

    def test_task_alternatives(self) -> None:
        assert _WIT_TYPE_ALTERNATIVES["Task"] == ["Task"]

    def test_bug_alternatives(self) -> None:
        assert _WIT_TYPE_ALTERNATIVES["Bug"] == ["Bug"]


# =============================================================================
# _resolve_work_item_type
# =============================================================================


class TestResolveWorkItemType:
    """Tests for the _resolve_work_item_type helper."""

    def test_user_story_found_in_project(self) -> None:
        """When project supports User Story, return it as-is."""
        result = _resolve_work_item_type("User Story", {"User Story", "Task", "Bug"})
        assert result == "User Story"

    def test_user_story_falls_back_to_issue(self) -> None:
        """When project lacks User Story but has Issue, fall back to Issue."""
        result = _resolve_work_item_type("User Story", {"Issue", "Task", "Bug"})
        assert result == "Issue"

    def test_epic_falls_back_to_feature(self) -> None:
        """When project lacks Epic but has Feature, fall back to Feature."""
        result = _resolve_work_item_type("Epic", {"Feature", "Task", "Bug"})
        assert result == "Feature"

    def test_epic_falls_back_to_issue_when_no_feature(self) -> None:
        """When project has neither Epic nor Feature, fall back to Issue."""
        result = _resolve_work_item_type("Epic", {"Issue", "Task", "Bug"})
        assert result == "Issue"

    def test_unknown_type_returns_itself(self) -> None:
        """Unknown types not in the alternatives dict return themselves."""
        result = _resolve_work_item_type("CustomType", {"CustomType", "Task"})
        assert result == "CustomType"

    def test_unknown_type_not_in_project_returns_itself(self) -> None:
        """Unknown types not in the project types still return the draft type."""
        result = _resolve_work_item_type("CustomType", {"Task", "Bug"})
        assert result == "CustomType"

    def test_empty_draft_type_defaults_to_user_story(self) -> None:
        """Empty string draft_type defaults to User Story."""
        result = _resolve_work_item_type("", {"User Story", "Task"})
        assert result == "User Story"

    def test_none_draft_type_defaults_to_user_story(self) -> None:
        """None draft_type defaults to User Story."""
        result = _resolve_work_item_type(None, {"User Story", "Task"})
        assert result == "User Story"

    def test_whitespace_draft_type_defaults_to_user_story(self) -> None:
        """Whitespace-only draft_type defaults to User Story."""
        result = _resolve_work_item_type("   ", {"User Story", "Task"})
        assert result == "User Story"

    def test_user_story_falls_back_to_product_backlog_item(self) -> None:
        """Scrum process: User Story -> Product Backlog Item."""
        result = _resolve_work_item_type("User Story", {"Product Backlog Item", "Task"})
        assert result == "Product Backlog Item"

    def test_fallback_returns_first_alternative_when_none_match(self) -> None:
        """When no alternatives are in the project, return the first alternative."""
        result = _resolve_work_item_type("User Story", {"Zephyr"})
        assert result == "User Story"


# =============================================================================
# PushToDevOpsTool — definition
# =============================================================================


class TestPushToDevOpsToolDefinition:
    """Tests for PushToDevOpsTool.definition property."""

    def test_name(self) -> None:
        tool = PushToDevOpsTool(session_manager=AsyncMock())
        assert tool.definition.name == "push_to_devops"

    def test_category(self) -> None:
        tool = PushToDevOpsTool(session_manager=AsyncMock())
        assert tool.definition.category == "push"

    def test_has_draft_ids_parameter(self) -> None:
        tool = PushToDevOpsTool(session_manager=AsyncMock())
        param_names = [p.name for p in tool.definition.parameters]
        assert "draft_ids" in param_names

    def test_has_area_and_iteration_params(self) -> None:
        tool = PushToDevOpsTool(session_manager=AsyncMock())
        param_names = [p.name for p in tool.definition.parameters]
        assert "area_path" in param_names
        assert "iteration_path" in param_names

    def test_has_additional_tags_param(self) -> None:
        tool = PushToDevOpsTool(session_manager=AsyncMock())
        param_names = [p.name for p in tool.definition.parameters]
        assert "additional_tags" in param_names


# =============================================================================
# PushToDevOpsTool — execute
# =============================================================================


class TestPushToDevOpsToolExecute:
    """Tests for PushToDevOpsTool.execute."""

    async def test_execute_success(self) -> None:
        """Successfully push a single draft to DevOps."""
        draft = _make_draft()
        session_manager = AsyncMock()
        session_manager.get_drafts.return_value = _make_draft_collection([draft])
        session_manager.delete_draft.return_value = None

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()
        mock_client.create_work_item.return_value = _mock_create_work_item_response()

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["ensure"], new_callable=AsyncMock, return_value=False),
            patch(paths["restore"], new_callable=AsyncMock, return_value=False),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushToDevOpsTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data["created"]) == 1
        assert result.data["created"][0]["work_item_id"] == 42
        session_manager.delete_draft.assert_awaited_once()

    async def test_execute_no_drafts_returns_failure(self) -> None:
        """When there are no drafts, return failure."""
        session_manager = AsyncMock()
        session_manager.get_drafts.return_value = _make_draft_collection([])

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["get_settings"], return_value=settings),
            patch(paths["ensure"], new_callable=AsyncMock, return_value=False),
            patch(paths["restore"], new_callable=AsyncMock, return_value=False),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            tool = PushToDevOpsTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.ERROR
        assert "No drafts" in result.error

    async def test_execute_with_parent_child_linking(self) -> None:
        """When drafts have parent_draft_id, child links to parent's ADO ID."""
        parent = _make_draft(draft_id="epic-001", work_item_type="Epic", title="Parent Epic")
        child = _make_draft(
            draft_id="story-001",
            work_item_type="User Story",
            title="Child Story",
            parent_draft_id="epic-001",
        )
        session_manager = AsyncMock()
        session_manager.get_drafts.return_value = _make_draft_collection([parent, child])
        session_manager.delete_draft.return_value = None

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()

        # Parent gets id 100, child gets id 101
        mock_client.create_work_item.side_effect = [
            _mock_create_work_item_response(100, "Parent Epic", "Epic"),
            _mock_create_work_item_response(101, "Child Story", "User Story"),
        ]
        mock_client.add_work_item_link.return_value = None

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["ensure"], new_callable=AsyncMock, return_value=False),
            patch(paths["restore"], new_callable=AsyncMock, return_value=False),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushToDevOpsTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data["created"]) == 2
        # Verify parent-child link was established
        mock_client.add_work_item_link.assert_awaited_once_with(
            project="Test Project",
            work_item_id=101,
            target_id=100,
            link_type="System.LinkTypes.Hierarchy-Reverse",
        )

    async def test_execute_additional_tags_deduplication(self) -> None:
        """Additional tags and default_push_tag are deduplicated with draft tags."""
        draft = _make_draft(tags=["mvp", "@AIGenerated"])
        session_manager = AsyncMock()
        session_manager.get_drafts.return_value = _make_draft_collection([draft])
        session_manager.delete_draft.return_value = None

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()
        mock_client.create_work_item.return_value = _mock_create_work_item_response()

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["ensure"], new_callable=AsyncMock, return_value=False),
            patch(paths["restore"], new_callable=AsyncMock, return_value=False),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushToDevOpsTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx, additional_tags=["mvp", "sprint-1"])

        assert result.status == ToolStatus.SUCCESS
        # Check that create_work_item was called with deduplicated tags
        call_kwargs = mock_client.create_work_item.call_args
        fields = call_kwargs.kwargs.get("fields") or call_kwargs[1].get("fields")
        tags_str = fields["System.Tags"]
        tag_list = [t.strip() for t in tags_str.split(";")]
        # @AIGenerated already in draft tags — should not duplicate
        assert tag_list.count("@AIGenerated") == 1
        assert tag_list.count("mvp") == 1
        assert "sprint-1" in tag_list

    async def test_execute_partial_failure(self) -> None:
        """When some items fail, return success with failed list populated."""
        from src.clients.devops import AzureDevOpsError

        good_draft = _make_draft(draft_id="good-001", title="Good Item")
        bad_draft = _make_draft(draft_id="bad-001", title="Bad Item")
        session_manager = AsyncMock()
        session_manager.get_drafts.return_value = _make_draft_collection([good_draft, bad_draft])
        session_manager.delete_draft.return_value = None

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()
        # First succeeds, second raises
        mock_client.create_work_item.side_effect = [
            _mock_create_work_item_response(42, "Good Item"),
            AzureDevOpsError("Validation error"),
        ]

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["ensure"], new_callable=AsyncMock, return_value=False),
            patch(paths["restore"], new_callable=AsyncMock, return_value=False),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushToDevOpsTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data["created"]) == 1
        assert len(result.data["failed"]) == 1
        assert result.data["failed"][0]["draft_id"] == "good-001"

    async def test_execute_all_fail_returns_failure(self) -> None:
        """When ALL items fail, return failure status."""
        from src.clients.devops import AzureDevOpsError

        draft = _make_draft()
        session_manager = AsyncMock()
        session_manager.get_drafts.return_value = _make_draft_collection([draft])

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()
        mock_client.create_work_item.side_effect = AzureDevOpsError("Boom")

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["ensure"], new_callable=AsyncMock, return_value=False),
            patch(paths["restore"], new_callable=AsyncMock, return_value=False),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushToDevOpsTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.ERROR
        assert "failed" in result.error.lower()

    async def test_area_path_and_iteration_path_passed_through(self) -> None:
        """Area and iteration paths from kwargs are applied to fields."""
        draft = _make_draft()
        session_manager = AsyncMock()
        session_manager.get_drafts.return_value = _make_draft_collection([draft])
        session_manager.delete_draft.return_value = None

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()
        mock_client.create_work_item.return_value = _mock_create_work_item_response()

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["ensure"], new_callable=AsyncMock, return_value=False),
            patch(paths["restore"], new_callable=AsyncMock, return_value=False),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushToDevOpsTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(
                ctx,
                area_path="MyProject\\Team A",
                iteration_path="MyProject\\Sprint 1",
            )

        assert result.status == ToolStatus.SUCCESS
        call_kwargs = mock_client.create_work_item.call_args
        fields = call_kwargs.kwargs.get("fields") or call_kwargs[1].get("fields")
        assert fields["System.AreaPath"] == "MyProject\\Team A"
        assert fields["System.IterationPath"] == "MyProject\\Sprint 1"

    async def test_custom_fields_applied(self) -> None:
        """Custom fields from the draft are included in the create call."""
        draft = _make_draft(custom_fields={"Custom.MyField": "hello"})
        session_manager = AsyncMock()
        session_manager.get_drafts.return_value = _make_draft_collection([draft])
        session_manager.delete_draft.return_value = None

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()
        mock_client.create_work_item.return_value = _mock_create_work_item_response()

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["ensure"], new_callable=AsyncMock, return_value=False),
            patch(paths["restore"], new_callable=AsyncMock, return_value=False),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushToDevOpsTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        call_kwargs = mock_client.create_work_item.call_args
        fields = call_kwargs.kwargs.get("fields") or call_kwargs[1].get("fields")
        assert fields["Custom.MyField"] == "hello"

    async def test_type_mapping_recorded_when_resolved_differs(self) -> None:
        """When resolved type differs from draft type, type_mapping is returned."""
        # Basic process: has Issue but not User Story
        draft = _make_draft(work_item_type="User Story")
        session_manager = AsyncMock()
        session_manager.get_drafts.return_value = _make_draft_collection([draft])
        session_manager.delete_draft.return_value = None

        # Project only has Issue, Task, Bug (no User Story)
        basic_types = []
        for name in ["Issue", "Task", "Bug"]:
            t = MagicMock()
            t.name = name
            basic_types.append(t)

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = basic_types
        mock_client.create_work_item.return_value = _mock_create_work_item_response(
            42, "Draft Title", "Issue"
        )

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["ensure"], new_callable=AsyncMock, return_value=False),
            patch(paths["restore"], new_callable=AsyncMock, return_value=False),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushToDevOpsTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert "type_mapping" in result.data
        assert result.data["type_mapping"]["User Story"] == "Issue"


# =============================================================================
# PushSingleTool — definition
# =============================================================================


class TestPushSingleToolDefinition:
    """Tests for PushSingleTool.definition property."""

    def test_name(self) -> None:
        tool = PushSingleTool(session_manager=AsyncMock())
        assert tool.definition.name == "push_single"

    def test_category(self) -> None:
        tool = PushSingleTool(session_manager=AsyncMock())
        assert tool.definition.category == "push"

    def test_draft_id_required(self) -> None:
        tool = PushSingleTool(session_manager=AsyncMock())
        draft_id_param = next(p for p in tool.definition.parameters if p.name == "draft_id")
        assert draft_id_param.required is True

    def test_has_parent_id_param(self) -> None:
        tool = PushSingleTool(session_manager=AsyncMock())
        param_names = [p.name for p in tool.definition.parameters]
        assert "parent_id" in param_names


# =============================================================================
# PushSingleTool — execute
# =============================================================================


class TestPushSingleToolExecute:
    """Tests for PushSingleTool.execute."""

    async def test_execute_success(self) -> None:
        """Successfully push a single draft."""
        draft = _make_draft()
        session_manager = AsyncMock()
        session_manager.get_draft.return_value = draft
        session_manager.delete_draft.return_value = None

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()
        mock_client.create_work_item.return_value = _mock_create_work_item_response()

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushSingleTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx, draft_id="draft-001")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["work_item_id"] == 42
        session_manager.delete_draft.assert_awaited_once()

    async def test_draft_not_found_returns_failure(self) -> None:
        """When the draft does not exist, return failure."""
        session_manager = AsyncMock()
        session_manager.get_draft.return_value = None

        tool = PushSingleTool(session_manager)
        ctx = _make_context()
        result = await tool.execute(ctx, draft_id="nonexistent")

        assert result.status == ToolStatus.ERROR
        assert "not found" in result.error.lower()

    async def test_with_parent_id_links_to_parent(self) -> None:
        """When parent_id is given, add_work_item_link is called."""
        draft = _make_draft()
        session_manager = AsyncMock()
        session_manager.get_draft.return_value = draft
        session_manager.delete_draft.return_value = None

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()
        mock_client.create_work_item.return_value = _mock_create_work_item_response(50)
        mock_client.add_work_item_link.return_value = None

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushSingleTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx, draft_id="draft-001", parent_id=999)

        assert result.status == ToolStatus.SUCCESS
        mock_client.add_work_item_link.assert_awaited_once_with(
            project="Test Project",
            work_item_id=50,
            target_id=999,
            link_type="System.LinkTypes.Hierarchy-Reverse",
        )

    async def test_story_points_mapped_to_effort_for_issue(self) -> None:
        """When resolved type is Issue, story_points maps to Effort field."""
        draft = _make_draft(work_item_type="User Story", story_points=8.0)
        session_manager = AsyncMock()
        session_manager.get_draft.return_value = draft
        session_manager.delete_draft.return_value = None

        # Project with only Issue (Basic process)
        issue_type = MagicMock()
        issue_type.name = "Issue"
        task_type = MagicMock()
        task_type.name = "Task"

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = [issue_type, task_type]
        mock_client.create_work_item.return_value = _mock_create_work_item_response(
            42, "Draft Title", "Issue"
        )

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushSingleTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx, draft_id="draft-001")

        assert result.status == ToolStatus.SUCCESS
        call_kwargs = mock_client.create_work_item.call_args
        fields = call_kwargs.kwargs.get("fields") or call_kwargs[1].get("fields")
        # Issue type uses Effort, not StoryPoints
        assert fields["Microsoft.VSTS.Scheduling.Effort"] == 8.0
        assert "Microsoft.VSTS.Scheduling.StoryPoints" not in fields

    async def test_azure_devops_error_returns_failure(self) -> None:
        """When AzureDevOpsClient raises, return failure."""
        from src.clients.devops import AzureDevOpsError

        draft = _make_draft()
        session_manager = AsyncMock()
        session_manager.get_draft.return_value = draft

        mock_client = AsyncMock()
        mock_client.get_work_item_types.return_value = _mock_work_item_types()
        mock_client.create_work_item.side_effect = AzureDevOpsError("Unauthorized")

        settings = _default_settings_mock()
        paths = _build_patchers()

        with (
            patch(paths["client_cls"]) as client_cls,
            patch(paths["get_settings"], return_value=settings),
            patch(paths["sync"], new_callable=AsyncMock),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = PushSingleTool(session_manager)
            ctx = _make_context()
            result = await tool.execute(ctx, draft_id="draft-001")

        assert result.status == ToolStatus.ERROR
        assert "Unauthorized" in result.error
