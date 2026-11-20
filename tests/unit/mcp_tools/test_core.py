"""Tests for src/mcp_tools/core.py -- core MCP tools for Azure DevOps."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_tools.base import (
    BaseTool,
    ToolContext,
    ToolResult,
    ToolStatus,
)
from src.mcp_tools.core import (
    GetUserProfileTool,
    GetWorkItemTypesTool,
    ListOrganizationsTool,
    ListProjectsTool,
    SelectProjectTool,
    create_core_tools,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def tool_context() -> ToolContext:
    """Authenticated tool context with org and project selected."""
    return ToolContext(
        session_id="sess-test",
        user_id="user-123",
        organization_url="https://dev.azure.com/TestOrg",
        project_id="TestProject",
        project_name="Test Project",
        access_token="test-token",
    )


@pytest.fixture()
def no_auth_context() -> ToolContext:
    """Context without access_token to test require_auth."""
    return ToolContext(
        session_id="sess-test",
        user_id="user-123",
        access_token=None,
    )


@pytest.fixture()
def no_project_context() -> ToolContext:
    """Context without project_id to test require_project."""
    return ToolContext(
        session_id="sess-test",
        user_id="user-123",
        access_token="test-token",
        project_id=None,
    )


@pytest.fixture()
def mock_settings() -> MagicMock:
    """Mock Settings returned by get_settings()."""
    settings = MagicMock()
    settings.azure_devops = MagicMock()
    settings.auth_enabled = True
    return settings


@pytest.fixture()
def mock_client() -> AsyncMock:
    """Mock AzureDevOpsClient as an async context manager."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# =============================================================================
# GetUserProfileTool
# =============================================================================


class TestGetUserProfileTool:
    """Tests for GetUserProfileTool."""

    def test_definition_properties(self) -> None:
        """Tool definition has correct name, category, and parameters."""
        tool = GetUserProfileTool()
        defn = tool.definition
        assert defn.name == "get_user_profile"
        assert defn.category == "core"
        assert defn.parameters == []

    async def test_execute_returns_user_info(self, tool_context: ToolContext) -> None:
        """Execute returns user profile data from context."""
        tool = GetUserProfileTool()
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["user_id"] == "user-123"
        assert result.data["organization_url"] == "https://dev.azure.com/TestOrg"
        assert result.data["project_id"] == "TestProject"
        assert result.data["project_name"] == "Test Project"

    async def test_require_auth_raises_without_token(
        self,
        no_auth_context: ToolContext,
    ) -> None:
        """Execute raises ValueError when not authenticated."""
        tool = GetUserProfileTool()
        with pytest.raises(ValueError, match="Not authenticated"):
            await tool.execute(no_auth_context)


# =============================================================================
# ListOrganizationsTool
# =============================================================================


class TestListOrganizationsTool:
    """Tests for ListOrganizationsTool."""

    def test_definition_properties(self) -> None:
        """Tool definition has correct name and category."""
        tool = ListOrganizationsTool()
        defn = tool.definition
        assert defn.name == "list_organizations"
        assert defn.category == "core"
        assert defn.parameters == []

    @patch("src.mcp_tools.core.get_settings")
    @patch("src.mcp_tools.core.AzureDevOpsClient")
    async def test_execute_success(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns list of organizations on success."""
        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client

        org = MagicMock()
        org.id = "org-1"
        org.name = "TestOrg"
        org.url = "https://dev.azure.com/TestOrg"
        mock_client.get_organizations.return_value = [org]

        tool = ListOrganizationsTool()
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data) == 1
        assert result.data[0]["id"] == "org-1"
        assert result.data[0]["name"] == "TestOrg"
        assert "1 organization" in result.message

    @patch("src.mcp_tools.core.get_settings")
    @patch("src.mcp_tools.core.AzureDevOpsClient")
    async def test_execute_failure_azure_error(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure on AzureDevOpsError."""
        from src.clients.devops import AzureDevOpsError

        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client
        mock_client.get_organizations.side_effect = AzureDevOpsError("API failure")

        tool = ListOrganizationsTool()
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.ERROR
        assert "API failure" in result.error

    async def test_require_auth_raises(self, no_auth_context: ToolContext) -> None:
        """Execute raises ValueError when not authenticated."""
        tool = ListOrganizationsTool()
        with pytest.raises(ValueError, match="Not authenticated"):
            await tool.execute(no_auth_context)


# =============================================================================
# ListProjectsTool
# =============================================================================


class TestListProjectsTool:
    """Tests for ListProjectsTool."""

    def test_definition_properties(self) -> None:
        """Tool definition has correct name, category, and optional org parameter."""
        tool = ListProjectsTool()
        defn = tool.definition
        assert defn.name == "list_projects"
        assert defn.category == "core"
        assert len(defn.parameters) == 1
        assert defn.parameters[0].name == "organization_url"
        assert defn.parameters[0].required is False

    @patch("src.mcp_tools.core.get_settings")
    @patch("src.mcp_tools.core.AzureDevOpsClient")
    async def test_execute_success(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns list of projects on success."""
        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client

        proj = MagicMock()
        proj.id = "proj-1"
        proj.name = "TestProject"
        proj.description = "A test project"
        proj.state = "wellFormed"
        proj.url = "https://dev.azure.com/TestOrg/TestProject"
        mock_client.get_projects.return_value = [proj]

        tool = ListProjectsTool()
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data) == 1
        assert result.data[0]["name"] == "TestProject"
        assert "1 project" in result.message

    async def test_execute_no_org_url_failure(self) -> None:
        """Execute returns failure when no organization_url is available."""
        ctx = ToolContext(
            session_id="s",
            user_id="u",
            access_token="tok",
            organization_url=None,
        )
        tool = ListProjectsTool()
        result = await tool.execute(ctx)

        assert result.status == ToolStatus.ERROR
        assert "No organization specified" in result.error

    @patch("src.mcp_tools.core.get_settings")
    @patch("src.mcp_tools.core.AzureDevOpsClient")
    async def test_execute_azure_error(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure on AzureDevOpsError."""
        from src.clients.devops import AzureDevOpsError

        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client
        mock_client.get_projects.side_effect = AzureDevOpsError("forbidden")

        tool = ListProjectsTool()
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.ERROR
        assert "forbidden" in result.error


# =============================================================================
# SelectProjectTool
# =============================================================================


class TestSelectProjectTool:
    """Tests for SelectProjectTool."""

    def test_definition_properties(self) -> None:
        """Tool definition has correct name, category, and parameters."""
        tool = SelectProjectTool()
        defn = tool.definition
        assert defn.name == "select_project"
        assert defn.category == "core"
        assert len(defn.parameters) == 2
        param_names = [p.name for p in defn.parameters]
        assert "project_id" in param_names
        assert "organization_url" in param_names

    @patch("src.mcp_tools.core.get_settings")
    @patch("src.mcp_tools.core.AzureDevOpsClient")
    async def test_execute_success(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns project details and update_context on success."""
        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client

        project = MagicMock()
        project.id = "proj-1"
        project.name = "TestProject"
        project.description = "A test project"
        project.state = "wellFormed"
        mock_client.get_project.return_value = project

        tool = SelectProjectTool()
        result = await tool.execute(tool_context, project_id="proj-1")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["id"] == "proj-1"
        assert result.data["name"] == "TestProject"
        assert result.metadata["update_context"]["project_id"] == "proj-1"
        assert "Selected project" in result.message

    @patch("src.mcp_tools.core.get_settings")
    @patch("src.mcp_tools.core.AzureDevOpsClient")
    async def test_execute_with_session_manager_stores_board_context(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """With a session_manager, board context is fetched and stored."""
        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client

        project = MagicMock()
        project.id = "proj-1"
        project.name = "TestProject"
        project.description = "A test project"
        project.state = "wellFormed"
        mock_client.get_project.return_value = project

        board_ctx = MagicMock()
        board_ctx.allowed_types = ["Epic", "User Story", "Task"]
        board_ctx.get_hierarchy_description.return_value = "Epic > User Story > Task"
        board_ctx.default_requirement_type = "User Story"
        mock_client.build_board_context.return_value = board_ctx

        mock_sm = AsyncMock()
        tool = SelectProjectTool(session_manager=mock_sm)
        result = await tool.execute(tool_context, project_id="proj-1")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["board_context"]["allowed_types"] == ["Epic", "User Story", "Task"]
        mock_sm.store_board_context.assert_awaited_once_with(
            session_id="sess-test",
            project_id="TestProject",
            board_context=board_ctx,
        )

    async def test_execute_no_org_url_failure(self) -> None:
        """Execute returns failure when no organization_url is available."""
        ctx = ToolContext(
            session_id="s",
            user_id="u",
            access_token="tok",
            organization_url=None,
        )
        tool = SelectProjectTool()
        result = await tool.execute(ctx, project_id="proj-1")

        assert result.status == ToolStatus.ERROR
        assert "No organization specified" in result.error

    @patch("src.mcp_tools.core.get_settings")
    @patch("src.mcp_tools.core.AzureDevOpsClient")
    async def test_execute_azure_error(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure on AzureDevOpsError."""
        from src.clients.devops import AzureDevOpsError

        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client
        mock_client.get_project.side_effect = AzureDevOpsError("not found")

        tool = SelectProjectTool()
        result = await tool.execute(tool_context, project_id="bad-id")

        assert result.status == ToolStatus.ERROR
        assert "not found" in result.error


# =============================================================================
# GetWorkItemTypesTool
# =============================================================================


class TestGetWorkItemTypesTool:
    """Tests for GetWorkItemTypesTool."""

    def test_definition_properties(self) -> None:
        """Tool definition has correct name and category."""
        tool = GetWorkItemTypesTool()
        defn = tool.definition
        assert defn.name == "get_work_item_types"
        assert defn.category == "core"
        assert defn.parameters == []

    @patch("src.mcp_tools.core.get_settings")
    @patch("src.mcp_tools.core.AzureDevOpsClient")
    async def test_execute_success(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns work item types on success."""
        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client

        wit = MagicMock()
        wit.name = "User Story"
        wit.reference_name = "Microsoft.VSTS.WorkItemTypes.UserStory"
        wit.description = "Tracks user stories"
        wit.color = "009CCC"
        mock_client.get_work_item_types.return_value = [wit]

        tool = GetWorkItemTypesTool()
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data) == 1
        assert result.data[0]["name"] == "User Story"
        assert result.data[0]["reference_name"] == "Microsoft.VSTS.WorkItemTypes.UserStory"
        assert "1 work item type" in result.message

    async def test_require_project_raises_without_project(
        self,
        no_project_context: ToolContext,
    ) -> None:
        """Execute raises ValueError when no project is selected."""
        tool = GetWorkItemTypesTool()
        with pytest.raises(ValueError, match="No project selected"):
            await tool.execute(no_project_context)

    @patch("src.mcp_tools.core.get_settings")
    @patch("src.mcp_tools.core.AzureDevOpsClient")
    async def test_execute_azure_error(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure on AzureDevOpsError."""
        from src.clients.devops import AzureDevOpsError

        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client
        mock_client.get_work_item_types.side_effect = AzureDevOpsError("no access")

        tool = GetWorkItemTypesTool()
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.ERROR
        assert "no access" in result.error


# =============================================================================
# create_core_tools factory
# =============================================================================


class TestCreateCoreTools:
    """Tests for the create_core_tools factory function."""

    def test_returns_five_tools(self) -> None:
        """Factory returns exactly 5 core tools."""
        tools = create_core_tools(session_manager=None)
        assert len(tools) == 5

    def test_all_are_base_tool_instances(self) -> None:
        """All returned objects are BaseTool subclasses."""
        tools = create_core_tools(session_manager=None)
        for tool in tools:
            assert isinstance(tool, BaseTool)

    def test_tool_names(self) -> None:
        """Factory creates all expected tool names."""
        tools = create_core_tools(session_manager=None)
        names = {t.definition.name for t in tools}
        expected = {
            "get_user_profile",
            "list_organizations",
            "list_projects",
            "select_project",
            "get_work_item_types",
        }
        assert names == expected

    def test_select_project_receives_session_manager(self) -> None:
        """SelectProjectTool gets the session_manager argument."""
        mock_sm = MagicMock()
        tools = create_core_tools(session_manager=mock_sm)
        select_tool = next(t for t in tools if t.definition.name == "select_project")
        assert isinstance(select_tool, SelectProjectTool)
        assert select_tool._session_manager is mock_sm

    def test_without_session_manager(self) -> None:
        """Factory works when session_manager is None."""
        tools = create_core_tools(session_manager=None)
        select_tool = next(t for t in tools if t.definition.name == "select_project")
        assert select_tool._session_manager is None
