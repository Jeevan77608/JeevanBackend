"""Tests for src/mcp_tools/board_wiki.py — Board and Wiki tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.devops import AzureDevOpsError
from src.mcp_tools.base import ToolContext, ToolResult, ToolStatus
from src.mcp_tools.board_wiki import (
    GetBacklogsTool,
    GetBoardStructureTool,
    ListWikiPagesTool,
    ListWikisTool,
    ReadWikiPageTool,
)


# =============================================================================
# Helpers
# =============================================================================

_PATCH_CLIENT = "src.mcp_tools.board_wiki.AzureDevOpsClient"
_PATCH_SETTINGS = "src.mcp_tools.board_wiki.get_settings"


def _make_context(**overrides: Any) -> ToolContext:
    """Build a ToolContext with all fields populated."""
    defaults = dict(
        session_id="sess-board-001",
        user_id="user-board-001",
        organization_url="https://dev.azure.com/TestOrg",
        project_id="TestProject",
        project_name="Test Project",
        access_token="test-access-token",
        conversation_id=None,
    )
    defaults.update(overrides)
    return ToolContext(**defaults)


def _default_settings() -> MagicMock:
    """Return a mock settings object for board/wiki tools."""
    settings = MagicMock()
    settings.azure_devops = MagicMock()
    settings.auth_enabled = False
    return settings


def _make_board(board_id: str = "board-1", name: str = "Stories", url: str = "") -> MagicMock:
    """Create a mock Board object."""
    board = MagicMock()
    board.id = board_id
    board.name = name
    board.url = url or f"https://dev.azure.com/TestOrg/_boards/{board_id}"
    return board


def _make_backlog(
    backlog_id: str = "bl-1",
    name: str = "Stories",
    rank: int = 1,
    work_item_types: list[str] | None = None,
) -> MagicMock:
    """Create a mock Backlog object."""
    backlog = MagicMock()
    backlog.id = backlog_id
    backlog.name = name
    backlog.rank = rank
    backlog.work_item_types = work_item_types or ["User Story"]
    return backlog


def _make_wiki(
    wiki_id: str = "wiki-1",
    name: str = "Project Wiki",
    wiki_type: str = "projectWiki",
    url: str = "",
) -> MagicMock:
    """Create a mock Wiki object."""
    wiki = MagicMock()
    wiki.id = wiki_id
    wiki.name = name
    wiki.type = wiki_type
    wiki.url = url or f"https://dev.azure.com/TestOrg/_wiki/{wiki_id}"
    return wiki


def _make_wiki_page(
    path: str = "/Home",
    content: str = "# Home Page",
    order: int | None = 0,
    url: str | None = None,
) -> MagicMock:
    """Create a mock WikiPage object."""
    page = MagicMock()
    page.path = path
    page.content = content
    page.order = order
    page.url = url or f"https://dev.azure.com/TestOrg/_wiki/pages{path}"
    return page


# =============================================================================
# GetBoardStructureTool — definition
# =============================================================================


class TestGetBoardStructureToolDefinition:
    """Tests for GetBoardStructureTool.definition."""

    def test_name(self) -> None:
        tool = GetBoardStructureTool()
        assert tool.definition.name == "get_board_structure"

    def test_category(self) -> None:
        tool = GetBoardStructureTool()
        assert tool.definition.category == "board"

    def test_board_name_is_optional(self) -> None:
        tool = GetBoardStructureTool()
        board_param = next(p for p in tool.definition.parameters if p.name == "board_name")
        assert board_param.required is False


# =============================================================================
# GetBoardStructureTool — execute
# =============================================================================


class TestGetBoardStructureToolExecute:
    """Tests for GetBoardStructureTool.execute."""

    async def test_no_board_name_lists_boards(self) -> None:
        """When no board_name is given, return list of all boards."""
        boards = [_make_board("b1", "Stories"), _make_board("b2", "Bugs")]

        mock_client = AsyncMock()
        mock_client.get_boards.return_value = boards

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = GetBoardStructureTool()
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data) == 2
        assert result.data[0]["name"] == "Stories"
        assert "2 board(s)" in result.message

    async def test_with_board_name_gets_columns(self) -> None:
        """When board_name is given, return columns for that board."""
        board = _make_board("b1", "Stories")
        columns = [
            {"name": "New", "stateMappings": {}},
            {"name": "Active", "stateMappings": {}},
            {"name": "Closed", "stateMappings": {}},
        ]

        mock_client = AsyncMock()
        mock_client.get_boards.return_value = [board]
        mock_client.get_board_columns.return_value = columns

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = GetBoardStructureTool()
            ctx = _make_context()
            result = await tool.execute(ctx, board_name="Stories")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["board"]["name"] == "Stories"
        assert len(result.data["columns"]) == 3
        assert "3 columns" in result.message

    async def test_board_not_found_returns_failure(self) -> None:
        """When the specified board does not exist, return failure."""
        mock_client = AsyncMock()
        mock_client.get_boards.return_value = [_make_board("b1", "Stories")]

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = GetBoardStructureTool()
            ctx = _make_context()
            result = await tool.execute(ctx, board_name="Nonexistent")

        assert result.status == ToolStatus.ERROR
        assert "not found" in result.error.lower()

    async def test_azure_devops_error_returns_failure(self) -> None:
        """When the client raises AzureDevOpsError, return failure."""
        mock_client = AsyncMock()
        mock_client.get_boards.side_effect = AzureDevOpsError("API timeout")

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = GetBoardStructureTool()
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.ERROR
        assert "API timeout" in result.error


# =============================================================================
# GetBacklogsTool — definition
# =============================================================================


class TestGetBacklogsToolDefinition:
    """Tests for GetBacklogsTool.definition."""

    def test_name(self) -> None:
        tool = GetBacklogsTool()
        assert tool.definition.name == "get_backlogs"

    def test_category(self) -> None:
        tool = GetBacklogsTool()
        assert tool.definition.category == "board"


# =============================================================================
# GetBacklogsTool — execute
# =============================================================================


class TestGetBacklogsToolExecute:
    """Tests for GetBacklogsTool.execute."""

    async def test_execute_success(self) -> None:
        """Return formatted backlog levels."""
        backlogs = [
            _make_backlog("bl-1", "Epics", 3, ["Epic"]),
            _make_backlog("bl-2", "Stories", 2, ["User Story"]),
            _make_backlog("bl-3", "Tasks", 1, ["Task"]),
        ]

        mock_client = AsyncMock()
        mock_client.get_backlogs.return_value = backlogs

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = GetBacklogsTool()
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data) == 3
        assert result.data[0]["name"] == "Epics"
        assert "3 backlog level(s)" in result.message

    async def test_azure_devops_error(self) -> None:
        """AzureDevOpsError is returned as failure."""
        mock_client = AsyncMock()
        mock_client.get_backlogs.side_effect = AzureDevOpsError("Forbidden")

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = GetBacklogsTool()
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.ERROR
        assert "Forbidden" in result.error


# =============================================================================
# ListWikisTool — definition
# =============================================================================


class TestListWikisToolDefinition:
    """Tests for ListWikisTool.definition."""

    def test_name(self) -> None:
        tool = ListWikisTool()
        assert tool.definition.name == "list_wikis"

    def test_category(self) -> None:
        tool = ListWikisTool()
        assert tool.definition.category == "wiki"

    def test_no_required_parameters(self) -> None:
        tool = ListWikisTool()
        assert tool.definition.parameters == []


# =============================================================================
# ListWikisTool — execute
# =============================================================================


class TestListWikisToolExecute:
    """Tests for ListWikisTool.execute."""

    async def test_execute_success(self) -> None:
        """Return formatted list of wikis."""
        wikis = [
            _make_wiki("wiki-1", "Project Wiki", "projectWiki"),
            _make_wiki("wiki-2", "Code Wiki", "codeWiki"),
        ]

        mock_client = AsyncMock()
        mock_client.get_wikis.return_value = wikis

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = ListWikisTool()
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data) == 2
        assert result.data[0]["name"] == "Project Wiki"
        assert result.data[1]["type"] == "codeWiki"
        assert "2 wiki(s)" in result.message

    async def test_azure_devops_error(self) -> None:
        """AzureDevOpsError is returned as failure."""
        mock_client = AsyncMock()
        mock_client.get_wikis.side_effect = AzureDevOpsError("Not found")

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = ListWikisTool()
            ctx = _make_context()
            result = await tool.execute(ctx)

        assert result.status == ToolStatus.ERROR
        assert "Not found" in result.error


# =============================================================================
# ReadWikiPageTool — definition
# =============================================================================


class TestReadWikiPageToolDefinition:
    """Tests for ReadWikiPageTool.definition."""

    def test_name(self) -> None:
        tool = ReadWikiPageTool()
        assert tool.definition.name == "read_wiki_page"

    def test_category(self) -> None:
        tool = ReadWikiPageTool()
        assert tool.definition.category == "wiki"

    def test_wiki_id_required(self) -> None:
        tool = ReadWikiPageTool()
        wiki_param = next(p for p in tool.definition.parameters if p.name == "wiki_id")
        assert wiki_param.required is True

    def test_page_path_required(self) -> None:
        tool = ReadWikiPageTool()
        path_param = next(p for p in tool.definition.parameters if p.name == "page_path")
        assert path_param.required is True


# =============================================================================
# ReadWikiPageTool — execute
# =============================================================================


class TestReadWikiPageToolExecute:
    """Tests for ReadWikiPageTool.execute."""

    async def test_execute_success(self) -> None:
        """Return wiki page content."""
        wiki_page = _make_wiki_page("/Requirements/Login", "# Login Requirements\n\nUsers must...")

        mock_client = AsyncMock()
        mock_client.get_wiki_page.return_value = wiki_page

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = ReadWikiPageTool()
            ctx = _make_context()
            result = await tool.execute(ctx, wiki_id="project.wiki", page_path="/Requirements/Login")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["path"] == "/Requirements/Login"
        assert "Login Requirements" in result.data["content"]

    async def test_azure_devops_error(self) -> None:
        """AzureDevOpsError is returned as failure."""
        mock_client = AsyncMock()
        mock_client.get_wiki_page.side_effect = AzureDevOpsError("Page not found")

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = ReadWikiPageTool()
            ctx = _make_context()
            result = await tool.execute(ctx, wiki_id="wiki-1", page_path="/Missing")

        assert result.status == ToolStatus.ERROR
        assert "Page not found" in result.error


# =============================================================================
# ListWikiPagesTool — definition
# =============================================================================


class TestListWikiPagesToolDefinition:
    """Tests for ListWikiPagesTool.definition."""

    def test_name(self) -> None:
        tool = ListWikiPagesTool()
        assert tool.definition.name == "list_wiki_pages"

    def test_category(self) -> None:
        tool = ListWikiPagesTool()
        assert tool.definition.category == "wiki"

    def test_wiki_id_required(self) -> None:
        tool = ListWikiPagesTool()
        wiki_param = next(p for p in tool.definition.parameters if p.name == "wiki_id")
        assert wiki_param.required is True

    def test_path_optional(self) -> None:
        tool = ListWikiPagesTool()
        path_param = next(p for p in tool.definition.parameters if p.name == "path")
        assert path_param.required is False


# =============================================================================
# ListWikiPagesTool — execute
# =============================================================================


class TestListWikiPagesToolExecute:
    """Tests for ListWikiPagesTool.execute."""

    async def test_execute_success_sorted_and_flattened(self) -> None:
        """Pages are sorted by order (None last) then path, and flattened."""
        page_a = _make_wiki_page("/Home", order=0)
        page_b = _make_wiki_page("/Requirements", order=1)
        page_c = _make_wiki_page("/Requirements/Login", order=None)
        page_d = _make_wiki_page("/About", order=2)

        mock_client = AsyncMock()
        # Return in unsorted order; tool should sort by (order, path)
        mock_client.list_wiki_pages.return_value = [page_c, page_d, page_a, page_b]

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = ListWikiPagesTool()
            ctx = _make_context()
            result = await tool.execute(ctx, wiki_id="project.wiki")

        assert result.status == ToolStatus.SUCCESS
        pages = result.data["pages"]
        assert result.data["total_count"] == 4
        # Sorted: order 0 (/Home), 1 (/Requirements), 2 (/About), None (/Requirements/Login)
        assert pages[0]["path"] == "/Home"
        assert pages[1]["path"] == "/Requirements"
        assert pages[2]["path"] == "/About"
        assert pages[3]["path"] == "/Requirements/Login"
        # Sequential numbering
        assert pages[0]["#"] == 1
        assert pages[3]["#"] == 4
        # Name extraction from path
        assert pages[0]["name"] == "Home"
        assert pages[3]["name"] == "Login"
        # Depth calculation
        assert pages[0]["depth"] == 0  # /Home -> depth 0
        assert pages[3]["depth"] == 1  # /Requirements/Login -> depth 1
        assert "4 page(s)" in result.message

    async def test_azure_devops_error(self) -> None:
        """AzureDevOpsError is returned as failure."""
        mock_client = AsyncMock()
        mock_client.list_wiki_pages.side_effect = AzureDevOpsError("Wiki does not exist")

        with (
            patch(_PATCH_CLIENT) as client_cls,
            patch(_PATCH_SETTINGS, return_value=_default_settings()),
        ):
            client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tool = ListWikiPagesTool()
            ctx = _make_context()
            result = await tool.execute(ctx, wiki_id="nonexistent")

        assert result.status == ToolStatus.ERROR
        assert "Wiki does not exist" in result.error
