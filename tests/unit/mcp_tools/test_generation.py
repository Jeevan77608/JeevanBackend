"""Tests for src/mcp_tools/generation.py — AI generation tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.devops import AzureDevOpsError
from src.clients.openai import (
    ContextLengthError,
    GenerationError,
    GenerationResult,
)
from src.mcp_tools.base import ToolContext, ToolStatus
from src.mcp_tools.generation import (
    GenerateBugReportTool,
    GenerateTasksTool,
    GenerateUserStoriesTool,
    RefineItemTool,
    add_draft_to_session,
    create_generation_tools,
    fetch_work_item_context,
)
from src.models.work_items import DraftCollection, DraftWorkItem


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def tool_context() -> ToolContext:
    """Fully-populated ToolContext for generation tests."""
    return ToolContext(
        session_id="sess-gen-001",
        user_id="user-gen-001",
        organization_url="https://dev.azure.com/TestOrg",
        project_id="TestProject",
        project_name="Test Project",
        access_token="test-access-token",
        conversation_id="conv-gen-001",
    )


@pytest.fixture()
def session_manager() -> AsyncMock:
    """Mock SessionManager with all draft-related methods."""
    mgr = AsyncMock()
    mgr.get_drafts.return_value = None
    mgr.save_drafts.return_value = None
    mgr.get_draft.return_value = None
    mgr.update_draft.return_value = None
    mgr.delete_draft.return_value = True
    mgr.get_board_context.return_value = None
    mgr.get_file_content.return_value = None
    return mgr


@pytest.fixture()
def sample_draft() -> DraftWorkItem:
    """A single draft work item for reuse across tests."""
    return DraftWorkItem(
        draft_id="d1",
        work_item_type="User Story",
        title="Test Story",
        description="A generated story",
        acceptance_criteria="AC 1",
        priority=2,
        story_points=3.0,
    )


@pytest.fixture()
def sample_generation_result(sample_draft: DraftWorkItem) -> GenerationResult:
    """Standard GenerationResult from the OpenAI client."""
    return GenerationResult(
        work_items=[sample_draft],
        prompt_tokens=100,
        completion_tokens=200,
        model="gpt-4",
    )


def _build_openai_mock(
    *,
    generate_user_stories: Any = None,
    generate_tasks: Any = None,
    generate_bug_report: Any = None,
    refine_work_item: Any = None,
    side_effect: Exception | None = None,
) -> AsyncMock:
    """Build an AzureOpenAIClient mock usable as an async context manager.

    Args:
        generate_user_stories: Return value for generate_user_stories.
        generate_tasks: Return value for generate_tasks.
        generate_bug_report: Return value for generate_bug_report.
        refine_work_item: Return value for refine_work_item.
        side_effect: If set, every method raises this exception.

    Returns:
        AsyncMock configured as async context manager.
    """
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    if side_effect:
        mock_client.generate_user_stories.side_effect = side_effect
        mock_client.generate_tasks.side_effect = side_effect
        mock_client.generate_bug_report.side_effect = side_effect
        mock_client.refine_work_item.side_effect = side_effect
    else:
        if generate_user_stories is not None:
            mock_client.generate_user_stories.return_value = generate_user_stories
        if generate_tasks is not None:
            mock_client.generate_tasks.return_value = generate_tasks
        if generate_bug_report is not None:
            mock_client.generate_bug_report.return_value = generate_bug_report
        if refine_work_item is not None:
            mock_client.refine_work_item.return_value = refine_work_item

    return mock_client


def _mock_settings() -> MagicMock:
    """Build a MagicMock that mimics get_settings() output."""
    settings = MagicMock()
    settings.auth_enabled = False
    settings.azure_devops = MagicMock()
    settings.azure_openai = MagicMock()
    return settings


# =============================================================================
# add_draft_to_session
# =============================================================================


class TestAddDraftToSession:
    """Tests for the add_draft_to_session helper."""

    async def test_creates_new_collection_when_none_exists(
        self, session_manager: AsyncMock, sample_draft: DraftWorkItem
    ) -> None:
        """When get_drafts returns None, a new DraftCollection is created and saved."""
        session_manager.get_drafts.return_value = None

        await add_draft_to_session(
            session_manager=session_manager,
            session_id="sess-1",
            project_id="proj-1",
            draft_item=sample_draft,
        )

        session_manager.save_drafts.assert_awaited_once()
        # save_drafts is called as save_drafts(session_id, drafts) — positional args
        call_args = session_manager.save_drafts.call_args[0]
        assert call_args[0] == "sess-1"
        saved_collection = call_args[1]
        assert isinstance(saved_collection, DraftCollection)
        assert saved_collection.session_id == "sess-1"
        assert saved_collection.project_id == "proj-1"
        assert len(saved_collection.items) == 1
        assert saved_collection.items[0].draft_id == "d1"

    async def test_appends_to_existing_collection(
        self, session_manager: AsyncMock, sample_draft: DraftWorkItem
    ) -> None:
        """When drafts already exist, new item is appended and collection is saved."""
        existing = DraftCollection(
            session_id="sess-1",
            project_id="proj-1",
            items=[
                DraftWorkItem(
                    draft_id="existing-1",
                    work_item_type="Task",
                    title="Existing Task",
                )
            ],
        )
        session_manager.get_drafts.return_value = existing

        await add_draft_to_session(
            session_manager=session_manager,
            session_id="sess-1",
            project_id="proj-1",
            draft_item=sample_draft,
        )

        session_manager.save_drafts.assert_awaited_once()
        saved_collection = session_manager.save_drafts.call_args[0][1]
        assert len(saved_collection.items) == 2
        assert saved_collection.items[1].draft_id == "d1"


# =============================================================================
# fetch_work_item_context
# =============================================================================


class TestFetchWorkItemContext:
    """Tests for the fetch_work_item_context helper."""

    async def test_empty_ids_returns_empty_string(
        self, tool_context: ToolContext
    ) -> None:
        """Passing no IDs should return empty string without making API calls."""
        result = await fetch_work_item_context(context=tool_context, work_item_ids=[])
        assert result == ""

    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureDevOpsClient")
    async def test_success_returns_context_strings(
        self,
        mock_devops_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
    ) -> None:
        """Successful fetch joins work item context strings."""
        mock_get_settings.return_value = _mock_settings()

        mock_wi_1 = MagicMock()
        mock_wi_1.to_context_string.return_value = "Work Item 101 context"
        mock_wi_2 = MagicMock()
        mock_wi_2.to_context_string.return_value = "Work Item 102 context"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_items_batch.return_value = [mock_wi_1, mock_wi_2]
        mock_devops_cls.return_value = mock_client

        result = await fetch_work_item_context(
            context=tool_context, work_item_ids=[101, 102]
        )

        assert "Work Item 101 context" in result
        assert "Work Item 102 context" in result
        assert "\n\n---\n\n" in result

    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureDevOpsClient")
    async def test_azure_devops_error_returns_empty_string(
        self,
        mock_devops_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
    ) -> None:
        """AzureDevOpsError is caught and returns empty string."""
        mock_get_settings.return_value = _mock_settings()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_items_batch.side_effect = AzureDevOpsError("API error")
        mock_devops_cls.return_value = mock_client

        result = await fetch_work_item_context(
            context=tool_context, work_item_ids=[999]
        )

        assert result == ""


# =============================================================================
# GenerateUserStoriesTool
# =============================================================================


class TestGenerateUserStoriesTool:
    """Tests for GenerateUserStoriesTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name, category, and required parameters."""
        tool = GenerateUserStoriesTool(session_manager)
        defn = tool.definition

        assert defn.name == "generate_user_stories"
        assert defn.category == "generation"
        param_names = [p.name for p in defn.parameters]
        assert "requirements" in param_names
        assert "additional_context" in param_names
        assert "file_id" in param_names
        assert "source_work_item_ids" in param_names
        assert "count" in param_names

        req_param = next(p for p in defn.parameters if p.name == "requirements")
        assert req_param.required is True

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_without_devops_token_and_with_explicit_count(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        sample_generation_result: GenerationResult,
    ) -> None:
        """Draft generation uses the signed-in session and honors explicit item counts."""
        mock_get_settings.return_value = _mock_settings()
        mock_openai = _build_openai_mock(
            generate_user_stories=sample_generation_result
        )
        mock_openai_cls.return_value = mock_openai
        context = ToolContext(
            session_id="sess-supabase-001",
            user_id="supabase:user",
            project_id="MDP",
            project_name="My discovery project",
            access_token=None,
        )

        result = await GenerateUserStoriesTool(session_manager).execute(
            context,
            requirements="Generate exactly 6 Epics.",
            count=6,
        )

        assert result.status == ToolStatus.SUCCESS
        mock_openai.generate_user_stories.assert_awaited_once_with(
            requirements="Generate exactly 6 Epics.",
            context="",
            board_context=None,
            count=6,
        )
        session_manager.save_drafts.assert_awaited()
        mock_sync.assert_awaited_once()

    async def test_execute_rejects_invalid_count(
        self,
        session_manager: AsyncMock,
    ) -> None:
        context = ToolContext(
            session_id="sess-gen-001",
            user_id="user-gen-001",
            project_id="MDP",
            access_token=None,
        )

        result = await GenerateUserStoriesTool(session_manager).execute(
            context,
            requirements="Generate Epics",
            count=0,
        )

        assert result.status == ToolStatus.ERROR
        assert result.error == "Count must be an integer between 1 and 10."

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_success(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_generation_result: GenerationResult,
    ) -> None:
        """Successful execution stores drafts and returns work items."""
        mock_get_settings.return_value = _mock_settings()
        mock_openai = _build_openai_mock(
            generate_user_stories=sample_generation_result
        )
        mock_openai_cls.return_value = mock_openai

        tool = GenerateUserStoriesTool(session_manager)
        result = await tool.execute(
            tool_context, requirements="Build a login page"
        )

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data["work_items"]) == 1
        assert result.data["work_items"][0]["draft_id"] == "d1"
        assert result.data["work_items"][0]["type"] == "User Story"
        assert result.data["usage"]["prompt_tokens"] == 100
        assert result.data["usage"]["model"] == "gpt-4"
        assert "Generated 1 user story" in result.message
        session_manager.save_drafts.assert_awaited()
        mock_sync.assert_awaited_once()

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_with_file_id_context(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_generation_result: GenerationResult,
    ) -> None:
        """When file_id is provided, file content is fetched and included."""
        mock_get_settings.return_value = _mock_settings()
        mock_openai = _build_openai_mock(
            generate_user_stories=sample_generation_result
        )
        mock_openai_cls.return_value = mock_openai
        session_manager.get_file_content.return_value = "File content here"

        tool = GenerateUserStoriesTool(session_manager)
        result = await tool.execute(
            tool_context,
            requirements="Build feature from spec",
            file_id="spec.pdf",
        )

        assert result.status == ToolStatus.SUCCESS
        session_manager.get_file_content.assert_awaited_once_with(
            session_id="sess-gen-001", file_id="spec.pdf"
        )

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.fetch_work_item_context", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_with_source_work_item_ids(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_fetch_ctx: AsyncMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_generation_result: GenerationResult,
    ) -> None:
        """When source_work_item_ids are provided, work item context is fetched."""
        mock_get_settings.return_value = _mock_settings()
        mock_fetch_ctx.return_value = "Epic #100 context"
        mock_openai = _build_openai_mock(
            generate_user_stories=sample_generation_result
        )
        mock_openai_cls.return_value = mock_openai

        tool = GenerateUserStoriesTool(session_manager)
        result = await tool.execute(
            tool_context,
            requirements="Implement auth",
            source_work_item_ids=[100],
        )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["context_source_ids"] == [100]
        mock_fetch_ctx.assert_awaited_once()

    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_context_length_error_returns_failure(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """ContextLengthError is caught and returned as failure."""
        mock_get_settings.return_value = _mock_settings()
        mock_openai = _build_openai_mock(
            side_effect=ContextLengthError(
                estimated_tokens=200_000, max_tokens=128_000
            )
        )
        mock_openai_cls.return_value = mock_openai

        tool = GenerateUserStoriesTool(session_manager)
        result = await tool.execute(
            tool_context, requirements="Huge requirements"
        )

        assert result.status == ToolStatus.ERROR
        assert "too large" in result.error

    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_generation_error_returns_failure(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """GenerationError is caught and returned as failure."""
        mock_get_settings.return_value = _mock_settings()
        mock_openai = _build_openai_mock(
            side_effect=GenerationError("JSON parse failed")
        )
        mock_openai_cls.return_value = mock_openai

        tool = GenerateUserStoriesTool(session_manager)
        result = await tool.execute(
            tool_context, requirements="Bad input"
        )

        assert result.status == ToolStatus.ERROR
        assert "Failed to generate user stories" in result.error


# =============================================================================
# GenerateTasksTool
# =============================================================================


class TestGenerateTasksTool:
    """Tests for GenerateTasksTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name, category, and parameters."""
        tool = GenerateTasksTool(session_manager)
        defn = tool.definition

        assert defn.name == "generate_tasks"
        assert defn.category == "generation"
        param_names = [p.name for p in defn.parameters]
        assert "user_story" in param_names
        assert "additional_context" in param_names
        assert "source_work_item_ids" in param_names

        story_param = next(p for p in defn.parameters if p.name == "user_story")
        assert story_param.required is True

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_with_draft_id_parent(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When user_story is a draft_id, the parent draft is looked up and tasks are linked."""
        mock_get_settings.return_value = _mock_settings()

        parent_draft = DraftWorkItem(
            draft_id="draft-parent-user-story-001",
            work_item_type="User Story",
            title="Parent Story",
            description="Parent description",
        )
        session_manager.get_draft.return_value = parent_draft

        task_item = DraftWorkItem(
            draft_id="d-task-1",
            work_item_type="Task",
            title="Implement API",
        )
        task_result = GenerationResult(
            work_items=[task_item],
            prompt_tokens=50,
            completion_tokens=100,
            model="gpt-4",
        )
        mock_openai = _build_openai_mock(generate_tasks=task_result)
        mock_openai_cls.return_value = mock_openai

        tool = GenerateTasksTool(session_manager)
        result = await tool.execute(
            tool_context,
            user_story="draft-parent-user-story-001",
        )

        assert result.status == ToolStatus.SUCCESS
        # Task should be linked to parent
        assert result.data["work_items"][0]["parent_draft_id"] == "draft-parent-user-story-001"
        assert "Generated 1 task" in result.message

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_with_title_match(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When user_story text matches a draft title, parent is resolved."""
        mock_get_settings.return_value = _mock_settings()

        # First get_draft returns None (not a draft ID), then get_drafts returns collection
        session_manager.get_draft.return_value = None

        story_draft = DraftWorkItem(
            draft_id="draft-story-match",
            work_item_type="User Story",
            title="Login Feature",
            description="Implement login",
        )
        collection = DraftCollection(
            session_id="sess-gen-001",
            project_id="TestProject",
            items=[story_draft],
        )
        session_manager.get_drafts.return_value = collection

        task_item = DraftWorkItem(
            draft_id="d-task-2",
            work_item_type="Task",
            title="Write login tests",
        )
        task_result = GenerationResult(
            work_items=[task_item],
            prompt_tokens=50,
            completion_tokens=80,
            model="gpt-4",
        )
        mock_openai = _build_openai_mock(generate_tasks=task_result)
        mock_openai_cls.return_value = mock_openai

        tool = GenerateTasksTool(session_manager)
        result = await tool.execute(
            tool_context, user_story="Login Feature"
        )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["work_items"][0]["parent_draft_id"] == "draft-story-match"

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_with_numeric_reference(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """Numeric reference like 'user story 2' resolves to the 2nd parent item."""
        mock_get_settings.return_value = _mock_settings()
        session_manager.get_draft.return_value = None

        story_1 = DraftWorkItem(
            draft_id="draft-s1",
            work_item_type="User Story",
            title="First Story",
        )
        story_2 = DraftWorkItem(
            draft_id="draft-s2",
            work_item_type="User Story",
            title="Second Story",
            description="The second one",
        )
        collection = DraftCollection(
            session_id="sess-gen-001",
            project_id="TestProject",
            items=[story_1, story_2],
        )
        session_manager.get_drafts.return_value = collection

        task_item = DraftWorkItem(
            draft_id="d-task-3",
            work_item_type="Task",
            title="Task for second story",
        )
        task_result = GenerationResult(
            work_items=[task_item],
            prompt_tokens=50,
            completion_tokens=80,
            model="gpt-4",
        )
        mock_openai = _build_openai_mock(generate_tasks=task_result)
        mock_openai_cls.return_value = mock_openai

        tool = GenerateTasksTool(session_manager)
        result = await tool.execute(
            tool_context, user_story="user story 2"
        )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["work_items"][0]["parent_draft_id"] == "draft-s2"

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_no_parent_found_uses_text_as_is(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When no matching parent draft is found, text is used directly."""
        mock_get_settings.return_value = _mock_settings()
        session_manager.get_draft.return_value = None
        session_manager.get_drafts.return_value = None

        task_item = DraftWorkItem(
            draft_id="d-task-4",
            work_item_type="Task",
            title="Standalone task",
        )
        task_result = GenerationResult(
            work_items=[task_item],
            prompt_tokens=30,
            completion_tokens=60,
            model="gpt-4",
        )
        mock_openai = _build_openai_mock(generate_tasks=task_result)
        mock_openai_cls.return_value = mock_openai

        tool = GenerateTasksTool(session_manager)
        result = await tool.execute(
            tool_context, user_story="Build something cool"
        )

        assert result.status == ToolStatus.SUCCESS
        # No parent linkage
        assert result.data["work_items"][0]["parent_draft_id"] is None


# =============================================================================
# GenerateBugReportTool
# =============================================================================


class TestGenerateBugReportTool:
    """Tests for GenerateBugReportTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name, category, and parameters."""
        tool = GenerateBugReportTool(session_manager)
        defn = tool.definition

        assert defn.name == "generate_bug_report"
        assert defn.category == "generation"
        param_names = [p.name for p in defn.parameters]
        assert "issue_description" in param_names
        assert "additional_context" in param_names
        assert "source_work_item_ids" in param_names

        issue_param = next(
            p for p in defn.parameters if p.name == "issue_description"
        )
        assert issue_param.required is True

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_success(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """Successful bug report generation stores draft and returns data."""
        mock_get_settings.return_value = _mock_settings()

        bug_item = DraftWorkItem(
            draft_id="d-bug-1",
            work_item_type="Bug",
            title="Login timeout error",
            description="Users experience timeout...",
            priority=1,
        )
        bug_result = GenerationResult(
            work_items=[bug_item],
            prompt_tokens=80,
            completion_tokens=150,
            model="gpt-4",
        )
        mock_openai = _build_openai_mock(generate_bug_report=bug_result)
        mock_openai_cls.return_value = mock_openai

        tool = GenerateBugReportTool(session_manager)
        result = await tool.execute(
            tool_context,
            issue_description="Users see timeout on login page",
        )

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data["work_items"]) == 1
        assert result.data["work_items"][0]["type"] == "Bug"
        assert result.data["work_items"][0]["priority"] == 1
        assert "Generated 1 bug report" in result.message
        session_manager.save_drafts.assert_awaited()
        mock_sync.assert_awaited_once()


# =============================================================================
# RefineItemTool
# =============================================================================


class TestRefineItemTool:
    """Tests for RefineItemTool definition and execute."""

    def test_definition_properties(self, session_manager: AsyncMock) -> None:
        """Definition has correct name, category, and parameters."""
        tool = RefineItemTool(session_manager)
        defn = tool.definition

        assert defn.name == "refine_item"
        assert defn.category == "generation"
        param_names = [p.name for p in defn.parameters]
        assert "draft_id" in param_names
        assert "feedback" in param_names

        for p in defn.parameters:
            assert p.required is True

    @patch("src.mcp_tools.generation.sync_drafts_to_conversation", new_callable=AsyncMock)
    @patch("src.mcp_tools.generation.get_settings")
    @patch("src.mcp_tools.generation.AzureOpenAIClient")
    async def test_execute_success(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_sync: AsyncMock,
        session_manager: AsyncMock,
        tool_context: ToolContext,
        sample_draft: DraftWorkItem,
    ) -> None:
        """Successful refinement updates draft and returns refined data."""
        mock_get_settings.return_value = _mock_settings()

        session_manager.get_draft.return_value = sample_draft

        refined_draft = MagicMock()
        refined_draft.draft_id = "d1"
        refined_draft.id = "d1"
        refined_draft.work_item_type = "User Story"
        refined_draft.title = "Test Story (Refined)"
        refined_draft.description = "Improved story"
        refined_draft.fields = {"System.Title": "Test Story (Refined)"}
        # refine_work_item returns a work item (with .id and .fields used by production code)
        mock_openai = _build_openai_mock(refine_work_item=refined_draft)
        mock_openai_cls.return_value = mock_openai

        tool = RefineItemTool(session_manager)
        result = await tool.execute(
            tool_context,
            draft_id="d1",
            feedback="Make it more specific",
        )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["title"] == "Test Story (Refined)"
        assert "Refined work item" in result.message
        session_manager.update_draft.assert_awaited_once()
        mock_sync.assert_awaited_once()

    async def test_execute_draft_not_found_returns_failure(
        self,
        session_manager: AsyncMock,
        tool_context: ToolContext,
    ) -> None:
        """When draft_id does not exist, returns failure."""
        session_manager.get_draft.return_value = None

        tool = RefineItemTool(session_manager)
        result = await tool.execute(
            tool_context,
            draft_id="nonexistent-draft",
            feedback="Improve it",
        )

        assert result.status == ToolStatus.ERROR
        assert "Draft not found" in result.error


# =============================================================================
# create_generation_tools Factory
# =============================================================================


class TestCreateGenerationTools:
    """Tests for the create_generation_tools factory function."""

    def test_returns_four_tools(self, session_manager: AsyncMock) -> None:
        """Factory returns all four generation tools."""
        tools = create_generation_tools(session_manager)

        assert len(tools) == 4
        names = {t.definition.name for t in tools}
        assert names == {
            "generate_user_stories",
            "generate_tasks",
            "generate_bug_report",
            "refine_item",
        }
