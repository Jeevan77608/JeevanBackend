"""Tests for src/mcp_tools/context.py -- work item context extraction tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_tools.base import (
    ToolContext,
    ToolResult,
    ToolStatus,
)
from src.mcp_tools.context import (
    GetWorkItemChildrenTool,
    GetWorkItemHierarchyTool,
    GetWorkItemTool,
    SearchWorkItemsTool,
    create_devops_client,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def tool_context() -> ToolContext:
    """Authenticated context with org and project selected."""
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
    """Context without access_token."""
    return ToolContext(
        session_id="sess-test",
        user_id="user-123",
        access_token=None,
    )


@pytest.fixture()
def no_project_context() -> ToolContext:
    """Context without project_id."""
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


def _make_work_item(
    *,
    id: int = 123,
    work_item_type: str = "User Story",
    title: str = "Test Story",
    state: str = "Active",
    description: str | None = "A test story",
    area_path: str = "TestProject\\Area",
    iteration_path: str = "TestProject\\Sprint1",
    assigned_to: str | None = "user@test.com",
    parent_id: int | None = None,
    child_ids: list[int] | None = None,
) -> MagicMock:
    """Create a mock WorkItem with the given attributes."""
    wi = MagicMock()
    wi.id = id
    wi.work_item_type = work_item_type
    wi.title = title
    wi.state = state
    wi.description = description
    wi.area_path = area_path
    wi.iteration_path = iteration_path
    wi.assigned_to = assigned_to
    wi.parent_id = parent_id
    wi.child_ids = child_ids or []
    wi.to_context_string.return_value = f"[{work_item_type} #{id}] {title}"
    return wi


# =============================================================================
# create_devops_client helper
# =============================================================================


class TestCreateDevopsClient:
    """Tests for the create_devops_client async context manager."""

    @patch("src.mcp_tools.context.get_settings")
    @patch("src.mcp_tools.context.AzureDevOpsClient")
    async def test_creates_client_with_correct_params(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """Client is created with settings, token, org_url, and use_pat flag."""
        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client

        async with create_devops_client(tool_context) as client:
            assert client is mock_client

        mock_client_cls.assert_called_once_with(
            settings=mock_settings.azure_devops,
            access_token="test-token",
            organization_url="https://dev.azure.com/TestOrg",
            use_pat=False,  # auth_enabled=True => use_pat=False
        )

    @patch("src.mcp_tools.context.get_settings")
    @patch("src.mcp_tools.context.AzureDevOpsClient")
    async def test_uses_pat_when_auth_disabled(
        self,
        mock_client_cls: MagicMock,
        mock_get_settings: MagicMock,
        tool_context: ToolContext,
        mock_settings: MagicMock,
        mock_client: AsyncMock,
    ) -> None:
        """When auth_enabled is False, use_pat is True."""
        mock_settings.auth_enabled = False
        mock_get_settings.return_value = mock_settings
        mock_client_cls.return_value = mock_client

        async with create_devops_client(tool_context) as client:
            pass

        _, kwargs = mock_client_cls.call_args
        assert kwargs["use_pat"] is True


# =============================================================================
# GetWorkItemTool
# =============================================================================


class TestGetWorkItemTool:
    """Tests for GetWorkItemTool."""

    def test_definition_properties(self) -> None:
        """Tool definition has correct name, category, and parameters."""
        tool = GetWorkItemTool()
        defn = tool.definition
        assert defn.name == "get_work_item"
        assert defn.category == "context"
        param_names = [p.name for p in defn.parameters]
        assert "work_item_id" in param_names
        assert "include_children" in param_names

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_success(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns work item details on success."""
        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)

        work_item = _make_work_item()
        mock_client.get_work_item.return_value = work_item

        tool = GetWorkItemTool()
        result = await tool.execute(tool_context, work_item_id=123)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["id"] == 123
        assert result.data["type"] == "User Story"
        assert result.data["title"] == "Test Story"
        assert result.data["state"] == "Active"
        assert result.data["description"] == "A test story"
        assert result.data["area_path"] == "TestProject\\Area"
        assert result.data["iteration_path"] == "TestProject\\Sprint1"
        assert result.data["assigned_to"] == "user@test.com"
        assert result.data["parent_id"] is None
        assert "context_string" in result.data
        assert "children" not in result.data
        assert "#123" in result.message

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_include_children(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """When include_children=True and child_ids exist, children are fetched."""
        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)

        parent = _make_work_item(id=100, work_item_type="Epic", title="Parent Epic", child_ids=[201, 202])
        child1 = _make_work_item(id=201, work_item_type="User Story", title="Child 1", state="New")
        child2 = _make_work_item(id=202, work_item_type="User Story", title="Child 2", state="Active")

        mock_client.get_work_item.return_value = parent
        mock_client.get_work_item_children.return_value = [child1, child2]

        tool = GetWorkItemTool()
        result = await tool.execute(tool_context, work_item_id=100, include_children=True)

        assert result.status == ToolStatus.SUCCESS
        assert "children" in result.data
        assert len(result.data["children"]) == 2
        assert result.data["children"][0]["id"] == 201
        assert result.data["children"][1]["id"] == 202

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_not_found(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure when work item is not found."""
        from src.clients.devops import NotFoundError

        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item.side_effect = NotFoundError("not found")

        tool = GetWorkItemTool()
        result = await tool.execute(tool_context, work_item_id=999)

        assert result.status == ToolStatus.ERROR
        assert "#999 not found" in result.error

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_azure_error(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure on AzureDevOpsError."""
        from src.clients.devops import AzureDevOpsError

        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item.side_effect = AzureDevOpsError("API error")

        tool = GetWorkItemTool()
        result = await tool.execute(tool_context, work_item_id=123)

        assert result.status == ToolStatus.ERROR
        assert "API error" in result.error

    async def test_require_auth_raises(self, no_auth_context: ToolContext) -> None:
        """Execute raises ValueError when not authenticated."""
        tool = GetWorkItemTool()
        with pytest.raises(ValueError, match="Not authenticated"):
            await tool.execute(no_auth_context, work_item_id=1)

    async def test_require_project_raises(self, no_project_context: ToolContext) -> None:
        """Execute raises ValueError when no project is selected."""
        tool = GetWorkItemTool()
        with pytest.raises(ValueError, match="No project selected"):
            await tool.execute(no_project_context, work_item_id=1)


# =============================================================================
# SearchWorkItemsTool
# =============================================================================


class TestSearchWorkItemsTool:
    """Tests for SearchWorkItemsTool."""

    def test_definition_properties(self) -> None:
        """Tool definition has correct name, category, and parameters."""
        tool = SearchWorkItemsTool()
        defn = tool.definition
        assert defn.name == "search_work_items"
        assert defn.category == "context"
        param_names = [p.name for p in defn.parameters]
        assert "work_item_type" in param_names
        assert "state" in param_names
        assert "title_contains" in param_names

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_success_with_filters(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns matching work items with filter params."""
        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)

        wi1 = _make_work_item(id=1, title="Login Feature")
        wi2 = _make_work_item(id=2, title="Logout Feature")
        mock_client.search_work_items.return_value = [wi1, wi2]

        tool = SearchWorkItemsTool()
        result = await tool.execute(
            tool_context,
            work_item_type="User Story",
            state="Active",
            title_contains="Feature",
        )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["count"] == 2
        assert len(result.data["work_items"]) == 2
        assert result.data["work_items"][0]["id"] == 1
        assert result.data["work_items"][1]["id"] == 2
        assert "2 work items" in result.message

        # Verify the client was called with the right filters
        mock_client.search_work_items.assert_awaited_once_with(
            project="Test Project",
            work_item_type="User Story",
            state="Active",
            title_contains="Feature",
            top=50,
        )

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_no_results(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns empty list when no work items match."""
        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.search_work_items.return_value = []

        tool = SearchWorkItemsTool()
        result = await tool.execute(tool_context, work_item_type="Epic")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["count"] == 0
        assert result.data["work_items"] == []
        assert "0 work items" in result.message

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_azure_error(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure on AzureDevOpsError."""
        from src.clients.devops import AzureDevOpsError

        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.search_work_items.side_effect = AzureDevOpsError("query failed")

        tool = SearchWorkItemsTool()
        result = await tool.execute(tool_context)

        assert result.status == ToolStatus.ERROR
        assert "query failed" in result.error


# =============================================================================
# GetWorkItemChildrenTool
# =============================================================================


class TestGetWorkItemChildrenTool:
    """Tests for GetWorkItemChildrenTool."""

    def test_definition_properties(self) -> None:
        """Tool definition has correct name, category, and parameters."""
        tool = GetWorkItemChildrenTool()
        defn = tool.definition
        assert defn.name == "get_work_item_children"
        assert defn.category == "context"
        assert len(defn.parameters) == 1
        assert defn.parameters[0].name == "parent_id"
        assert defn.parameters[0].required is True

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_success(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns parent info and children list."""
        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)

        parent = _make_work_item(id=100, work_item_type="Epic", title="Parent Epic")
        child1 = _make_work_item(id=201, title="Child Story 1", description="Short desc")
        child2 = _make_work_item(id=202, title="Child Story 2", description=None)

        mock_client.get_work_item.return_value = parent
        mock_client.get_work_item_children.return_value = [child1, child2]

        tool = GetWorkItemChildrenTool()
        result = await tool.execute(tool_context, parent_id=100)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["parent"]["id"] == 100
        assert result.data["parent"]["type"] == "Epic"
        assert result.data["count"] == 2
        assert len(result.data["children"]) == 2
        assert result.data["children"][0]["id"] == 201
        assert result.data["children"][0]["context_string"] is not None
        assert result.data["children"][1]["description"] is None
        assert "2 children" in result.message

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_not_found(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure when parent is not found."""
        from src.clients.devops import NotFoundError

        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item.side_effect = NotFoundError("not found")

        tool = GetWorkItemChildrenTool()
        result = await tool.execute(tool_context, parent_id=999)

        assert result.status == ToolStatus.ERROR
        assert "#999 not found" in result.error

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_no_children(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns empty children list when parent has no children."""
        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)

        parent = _make_work_item(id=100, work_item_type="Epic", title="Lonely Epic")
        mock_client.get_work_item.return_value = parent
        mock_client.get_work_item_children.return_value = []

        tool = GetWorkItemChildrenTool()
        result = await tool.execute(tool_context, parent_id=100)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["count"] == 0
        assert result.data["children"] == []
        assert "0 children" in result.message

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_azure_error(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure on AzureDevOpsError."""
        from src.clients.devops import AzureDevOpsError

        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item.side_effect = AzureDevOpsError("server error")

        tool = GetWorkItemChildrenTool()
        result = await tool.execute(tool_context, parent_id=100)

        assert result.status == ToolStatus.ERROR
        assert "server error" in result.error


# =============================================================================
# GetWorkItemHierarchyTool
# =============================================================================


class TestGetWorkItemHierarchyTool:
    """Tests for GetWorkItemHierarchyTool."""

    def test_definition_properties(self) -> None:
        """Tool definition has correct name, category, and parameters."""
        tool = GetWorkItemHierarchyTool()
        defn = tool.definition
        assert defn.name == "get_work_item_hierarchy"
        assert defn.category == "context"
        assert len(defn.parameters) == 1
        assert defn.parameters[0].name == "work_item_id"
        assert defn.parameters[0].required is True

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_success_with_parent_and_children(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns full hierarchy with parent and children."""
        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)

        work_item = _make_work_item(id=150, work_item_type="User Story", title="Middle Story")
        parent = _make_work_item(id=100, work_item_type="Epic", title="Parent Epic")
        child1 = _make_work_item(id=201, work_item_type="Task", title="Task 1", state="New")
        child2 = _make_work_item(id=202, work_item_type="Task", title="Task 2", state="Active")

        mock_client.get_work_item_hierarchy.return_value = {
            "work_item": work_item,
            "parent": parent,
            "children": [child1, child2],
        }

        tool = GetWorkItemHierarchyTool()
        result = await tool.execute(tool_context, work_item_id=150)

        assert result.status == ToolStatus.SUCCESS
        # Work item
        assert result.data["work_item"]["id"] == 150
        assert result.data["work_item"]["type"] == "User Story"
        assert result.data["work_item"]["title"] == "Middle Story"
        assert "context_string" in result.data["work_item"]
        # Parent
        assert result.data["parent"]["id"] == 100
        assert result.data["parent"]["type"] == "Epic"
        # Children
        assert len(result.data["children"]) == 2
        assert result.data["children"][0]["id"] == 201
        assert result.data["children"][1]["id"] == 202
        assert "hierarchy" in result.message.lower()

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_no_parent_no_children(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute handles work item with no parent and no children."""
        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)

        work_item = _make_work_item(id=150, work_item_type="Bug", title="Standalone Bug")

        mock_client.get_work_item_hierarchy.return_value = {
            "work_item": work_item,
            "parent": None,
            "children": [],
        }

        tool = GetWorkItemHierarchyTool()
        result = await tool.execute(tool_context, work_item_id=150)

        assert result.status == ToolStatus.SUCCESS
        assert result.data["parent"] is None
        assert result.data["children"] == []

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_not_found(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure when work item is not found."""
        from src.clients.devops import NotFoundError

        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item_hierarchy.side_effect = NotFoundError("not found")

        tool = GetWorkItemHierarchyTool()
        result = await tool.execute(tool_context, work_item_id=999)

        assert result.status == ToolStatus.ERROR
        assert "#999 not found" in result.error

    @patch("src.mcp_tools.context.create_devops_client")
    async def test_execute_azure_error(
        self,
        mock_create_client: MagicMock,
        tool_context: ToolContext,
        mock_client: AsyncMock,
    ) -> None:
        """Execute returns failure on AzureDevOpsError."""
        from src.clients.devops import AzureDevOpsError

        mock_create_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_create_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_work_item_hierarchy.side_effect = AzureDevOpsError("timeout")

        tool = GetWorkItemHierarchyTool()
        result = await tool.execute(tool_context, work_item_id=150)

        assert result.status == ToolStatus.ERROR
        assert "timeout" in result.error

    async def test_require_auth_raises(self, no_auth_context: ToolContext) -> None:
        """Execute raises ValueError when not authenticated."""
        tool = GetWorkItemHierarchyTool()
        with pytest.raises(ValueError, match="Not authenticated"):
            await tool.execute(no_auth_context, work_item_id=1)

    async def test_require_project_raises(self, no_project_context: ToolContext) -> None:
        """Execute raises ValueError when no project is selected."""
        tool = GetWorkItemHierarchyTool()
        with pytest.raises(ValueError, match="No project selected"):
            await tool.execute(no_project_context, work_item_id=1)
