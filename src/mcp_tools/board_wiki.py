"""
Board and Wiki MCP Tools for Azure DevOps.

This module provides tools for accessing Azure DevOps boards and wikis.
These tools help AI assistants understand project structure and access
documentation for context-aware work item generation.

Tools:
    - get_board_structure: Get board columns, swimlanes, and WIP limits
    - get_backlogs: Get backlog levels and their work item types
    - list_wikis: List all wikis in the project
    - read_wiki_page: Read wiki page content (Markdown)

Use Cases:
    - Understanding team workflow (board columns)
    - Identifying backlog hierarchy (Epic → Feature → Story → Task)
    - Accessing project documentation for context
    - Reading requirements from wiki pages

Example:
    # Get board structure to understand workflow
    get_board_structure(board_name="Stories")

    # List available wikis
    list_wikis()

    # Read requirements from wiki
    read_wiki_page(wiki_id="project.wiki", page_path="/Requirements/Sprint-1")
"""

from __future__ import annotations

import logging
from typing import Any

from src.clients.devops import AzureDevOpsClient, AzureDevOpsError
from src.config.settings import get_settings
from src.mcp_tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from src.mcp_tools.registry import register_stateless_tools

logger = logging.getLogger(__name__)


# =============================================================================
# Get Board Structure Tool
# =============================================================================


class GetBoardStructureTool(BaseTool):
    """Get the board structure including columns."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_board_structure",
            description="Get the board structure including columns and their configuration for the current project.",
            parameters=[
                ToolParameter(
                    name="board_name",
                    type="string",
                    description="Board name (e.g., 'Stories', 'Backlog Items'). If not specified, lists all boards.",
                    required=False,
                ),
                ToolParameter(
                    name="team",
                    type="string",
                    description="Team name. Uses default team if not specified.",
                    required=False,
                ),
            ],
            returns="Board structure with columns and their states",
            category="board",
            examples=[
                "Show me the board structure",
                "What columns are on the Stories board?",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute get_board_structure."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        board_name = params.get("board_name")
        team = params.get("team")

        settings = get_settings()
        use_pat = not settings.auth_enabled

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
                organization_url=context.organization_url,
                use_pat=use_pat,
            ) as client:
                boards = await client.get_boards(
                    project=context.project_name,
                    team=team,
                )

                if board_name:
                    # Find specific board and get columns
                    board = next((b for b in boards if b.name == board_name), None)
                    if not board:
                        return ToolResult.failure(f"Board '{board_name}' not found")

                    columns = await client.get_board_columns(
                        project=context.project_name,
                        board_id=board.id,
                        team=team,
                    )

                    return ToolResult.success(
                        data={
                            "board": {"id": board.id, "name": board.name},
                            "columns": columns,
                        },
                        message=f"Board '{board_name}' has {len(columns)} columns",
                    )
                else:
                    # Return list of boards
                    return ToolResult.success(
                        data=[
                            {"id": b.id, "name": b.name, "url": b.url}
                            for b in boards
                        ],
                        message=f"Found {len(boards)} board(s)",
                    )

        except AzureDevOpsError as e:
            logger.error(f"Failed to get board structure: {e}")
            return ToolResult.failure(str(e))


# =============================================================================
# Get Backlogs Tool
# =============================================================================


class GetBacklogsTool(BaseTool):
    """Get backlog levels for the project."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_backlogs",
            description="Get the backlog levels (Epic, Feature, Story, etc.) configured for the project.",
            parameters=[
                ToolParameter(
                    name="team",
                    type="string",
                    description="Team name. Uses default team if not specified.",
                    required=False,
                ),
            ],
            returns="List of backlog levels with their configuration",
            category="board",
            examples=[
                "What backlog levels are available?",
                "Show me the backlog hierarchy",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute get_backlogs."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        team = params.get("team")

        settings = get_settings()
        use_pat = not settings.auth_enabled

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
                organization_url=context.organization_url,
                use_pat=use_pat,
            ) as client:
                backlogs = await client.get_backlogs(
                    project=context.project_name,
                    team=team,
                )

                return ToolResult.success(
                    data=[
                        {
                            "id": b.id,
                            "name": b.name,
                            "rank": b.rank,
                            "work_item_types": b.work_item_types,
                        }
                        for b in backlogs
                    ],
                    message=f"Found {len(backlogs)} backlog level(s)",
                )

        except AzureDevOpsError as e:
            logger.error(f"Failed to get backlogs: {e}")
            return ToolResult.failure(str(e))


# =============================================================================
# List Wikis Tool
# =============================================================================


class ListWikisTool(BaseTool):
    """List wikis in the project."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_wikis",
            description="List all wikis available in the current project.",
            parameters=[],
            returns="List of wikis with id, name, and type",
            category="wiki",
            examples=[
                "Show me the project wikis",
                "What wikis are available?",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute list_wikis."""
        context.require_auth()
        context.require_project()

        settings = get_settings()
        use_pat = not settings.auth_enabled

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
                organization_url=context.organization_url,
                use_pat=use_pat,
            ) as client:
                wikis = await client.get_wikis(project=context.project_name)

            return ToolResult.success(
                data=[
                    {
                        "id": w.id,
                        "name": w.name,
                        "type": w.type,
                        "url": w.url,
                    }
                    for w in wikis
                ],
                message=f"Found {len(wikis)} wiki(s)",
            )

        except AzureDevOpsError as e:
            logger.error(f"Failed to list wikis: {e}")
            return ToolResult.failure(str(e))


# =============================================================================
# Read Wiki Page Tool
# =============================================================================


class ReadWikiPageTool(BaseTool):
    """Read a wiki page content."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_wiki_page",
            description="Read the content of a wiki page. Useful for understanding project context and requirements.",
            parameters=[
                ToolParameter(
                    name="wiki_id",
                    type="string",
                    description="Wiki ID or name",
                    required=True,
                ),
                ToolParameter(
                    name="page_path",
                    type="string",
                    description="Path to the wiki page (e.g., '/Home', '/Requirements/Feature-X')",
                    required=True,
                ),
            ],
            returns="Wiki page content in markdown format",
            category="wiki",
            examples=[
                "Read the Home page of the project wiki",
                "Show me the requirements page",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute read_wiki_page."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        wiki_id = params["wiki_id"]
        page_path = params["page_path"]

        settings = get_settings()
        use_pat = not settings.auth_enabled

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
                organization_url=context.organization_url,
                use_pat=use_pat,
            ) as client:
                wiki_page = await client.get_wiki_page(
                    project=context.project_name,
                    wiki_id=wiki_id,
                    path=page_path,
                )
                content = wiki_page.content or ""

            return ToolResult.success(
                data={
                    "path": page_path,
                    "content": content,
                },
                message=f"Retrieved wiki page: {page_path}",
            )

        except AzureDevOpsError as e:
            logger.error(f"Failed to read wiki page: {e}")
            return ToolResult.failure(str(e))


# =============================================================================
# List Wiki Pages Tool
# =============================================================================


class ListWikiPagesTool(BaseTool):
    """List all pages in a wiki."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_wiki_pages",
            description="List all pages in a wiki. Returns a flat list of all pages with their paths and URLs.",
            parameters=[
                ToolParameter(
                    name="wiki_id",
                    type="string",
                    description="Wiki ID or name (e.g., 'ProjectName.wiki')",
                    required=True,
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Root path to start listing from (default: '/')",
                    required=False,
                ),
            ],
            returns="List of wiki pages with paths and URLs",
            category="wiki",
            examples=[
                "List all pages in the project wiki",
                "Show me all wiki pages",
                "What pages are in the wiki?",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute list_wiki_pages."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        wiki_id = params["wiki_id"]
        # Ensure path defaults to "/" if not provided or empty
        path = params.get("path") or "/"

        settings = get_settings()
        use_pat = not settings.auth_enabled

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
                organization_url=context.organization_url,
                use_pat=use_pat,
            ) as client:
                pages = await client.list_wiki_pages(
                    project=context.project_name,
                    wiki_id=wiki_id,
                    path=path,
                )

            # Sort pages: by order (None last), then by path alphabetically
            sorted_pages = sorted(
                pages,
                key=lambda p: (p.order if p.order is not None else 999999, p.path),
            )

            # Format pages with sequential numbering
            page_list = []
            for idx, page in enumerate(sorted_pages, start=1):
                # Extract readable name from path (e.g., "/Requirements/Login" → "Login")
                name = page.path.rstrip("/").rsplit("/", 1)[-1] if page.path else ""
                # Calculate depth for hierarchy indication
                depth = page.path.count("/") - 1 if page.path.startswith("/") else page.path.count("/")
                page_list.append({
                    "#": idx,
                    "name": name,
                    "path": page.path,
                    "depth": depth,
                    "order": page.order,
                    "url": page.url,
                })

            return ToolResult.success(
                data={
                    "wiki_id": wiki_id,
                    "pages": page_list,
                    "total_count": len(page_list),
                },
                message=f"Found {len(page_list)} page(s) in wiki '{wiki_id}'",
            )

        except AzureDevOpsError as e:
            logger.error(f"Failed to list wiki pages: {e}")
            return ToolResult.failure(str(e))


# =============================================================================
# Tool Registry
# =============================================================================


BOARD_WIKI_TOOLS: list[BaseTool] = register_stateless_tools([
    GetBoardStructureTool(),
    GetBacklogsTool(),
    ListWikisTool(),
    ListWikiPagesTool(),
    ReadWikiPageTool(),
])
