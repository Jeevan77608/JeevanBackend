"""
MCP Tools for Work Item Context Extraction.

This module provides tools for fetching existing Azure DevOps work items
to use as context when generating new work items with AI. Context-aware
generation produces higher quality results by understanding:

- Project terminology and naming conventions
- Existing acceptance criteria patterns
- Story point estimation standards
- Team-specific work item structures

Tools:
    - get_work_item: Fetch a single work item by ID
    - search_work_items: Search work items with filters (type, state, query)
    - get_work_item_children: Get child items of a parent work item
    - get_work_item_hierarchy: Get full hierarchy tree (parent → children → grandchildren)

Usage Pattern:
    1. Use search_work_items to find relevant existing items
    2. Use get_work_item_hierarchy to understand structure
    3. Pass work item IDs to generation tools via source_work_item_ids

Example:
    # Find existing user stories for context
    search_work_items(work_item_type="User Story", state="Active")

    # Get hierarchy of an epic
    get_work_item_hierarchy(work_item_id=1001)

    # Generate new stories using existing ones as context
    generate_user_stories(
        requirements="Add payment processing",
        source_work_item_ids=[1001, 1002, 1003]
    )
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from src.clients.devops import AzureDevOpsClient, AzureDevOpsError, NotFoundError
from src.config.settings import get_settings
from src.mcp_tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from src.mcp_tools.registry import register_stateless_tools
from src.models.devops import WorkItem

logger = logging.getLogger(__name__)


@asynccontextmanager
async def create_devops_client(
    context: ToolContext,
) -> AsyncGenerator[AzureDevOpsClient, None]:
    """
    Create an Azure DevOps client with proper authentication.

    Uses PAT when OAuth is disabled (local development),
    Bearer token when OAuth is enabled (production).
    """
    settings = get_settings()
    use_pat = not settings.auth_enabled

    async with AzureDevOpsClient(
        settings=settings.azure_devops,
        access_token=context.access_token,
        organization_url=context.organization_url,
        use_pat=use_pat,
    ) as client:
        yield client


# =============================================================================
# Get Work Item Tool
# =============================================================================


class GetWorkItemTool(BaseTool):
    """Get a specific work item by ID for context."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_work_item",
            description=(
                "Get a specific work item by ID. Use this to fetch an Epic, User Story, "
                "Task, or Bug to use as context for generating related work items."
            ),
            parameters=[
                ToolParameter(
                    name="work_item_id",
                    type="number",
                    description="The ID of the work item to fetch",
                    required=True,
                ),
                ToolParameter(
                    name="include_children",
                    type="boolean",
                    description="Whether to include child work items",
                    required=False,
                    default=False,
                ),
            ],
            returns="Work item details including title, description, state, and optionally children",
            examples=[
                "get_work_item(work_item_id=123)",
                "get_work_item(work_item_id=456, include_children=True)",
            ],
            category="context",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        """Execute get_work_item tool."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        work_item_id = int(params["work_item_id"])
        include_children = params.get("include_children", False)

        try:
            async with create_devops_client(context) as client:
                work_item = await client.get_work_item(
                    work_item_id=work_item_id,
                    project=context.project_name,
                )

                result_data = {
                    "id": work_item.id,
                    "type": work_item.work_item_type,
                    "title": work_item.title,
                    "description": work_item.description,
                    "state": work_item.state,
                    "area_path": work_item.area_path,
                    "iteration_path": work_item.iteration_path,
                    "assigned_to": work_item.assigned_to,
                    "parent_id": work_item.parent_id,
                    "context_string": work_item.to_context_string(),
                }

                if include_children and work_item.child_ids:
                    children = await client.get_work_item_children(
                        work_item_id=work_item_id,
                        project=context.project_name,
                    )
                    result_data["children"] = [
                        {
                            "id": child.id,
                            "type": child.work_item_type,
                            "title": child.title,
                            "state": child.state,
                        }
                        for child in children
                    ]

                return ToolResult.success(
                    data=result_data,
                    message=f"Retrieved work item #{work_item_id}: {work_item.title}",
                )

        except NotFoundError:
            return ToolResult.failure(
                error=f"Work item #{work_item_id} not found",
            )
        except AzureDevOpsError as e:
            logger.error(f"Failed to get work item: {e}")
            return ToolResult.failure(error=str(e))


# =============================================================================
# Search Work Items Tool
# =============================================================================


class SearchWorkItemsTool(BaseTool):
    """Search for work items to use as context."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_work_items",
            description=(
                "Search for work items by type, state, area, or title. "
                "Use this to find existing Epics, User Stories, or Tasks to use as context."
            ),
            parameters=[
                ToolParameter(
                    name="work_item_type",
                    type="string",
                    description="Filter by work item type",
                    required=False,
                    enum=["Epic", "Feature", "User Story", "Task", "Bug", "Issue"],
                ),
                ToolParameter(
                    name="state",
                    type="string",
                    description="Filter by state (e.g., New, Active, Closed)",
                    required=False,
                ),
                ToolParameter(
                    name="title_contains",
                    type="string",
                    description="Filter by title containing this text",
                    required=False,
                ),
            ],
            returns="List of matching work items",
            examples=[
                "search_work_items(work_item_type='Epic')",
                "search_work_items(work_item_type='User Story', state='Active')",
                "search_work_items(title_contains='authentication')",
            ],
            category="context",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        """Execute search_work_items tool."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        work_item_type = params.get("work_item_type")
        state = params.get("state")
        title_contains = params.get("title_contains")

        try:
            async with create_devops_client(context) as client:
                work_items = await client.search_work_items(
                    project=context.project_name,
                    work_item_type=work_item_type,
                    state=state,
                    title_contains=title_contains,
                    top=50,
                )

                results = [
                    {
                        "id": wi.id,
                        "type": wi.work_item_type,
                        "title": wi.title,
                        "state": wi.state,
                        "assigned_to": wi.assigned_to,
                    }
                    for wi in work_items
                ]

                return ToolResult.success(
                    data={"work_items": results, "count": len(results)},
                    message=f"Found {len(results)} work items",
                )

        except AzureDevOpsError as e:
            logger.error(f"Failed to search work items: {e}")
            return ToolResult.failure(error=str(e))


# =============================================================================
# Get Work Item Children Tool
# =============================================================================


class GetWorkItemChildrenTool(BaseTool):
    """Get child work items of a parent (Epic -> Stories, Story -> Tasks)."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_work_item_children",
            description=(
                "Get all child work items of a parent. For example, get all User Stories "
                "under an Epic, or all Tasks under a User Story."
            ),
            parameters=[
                ToolParameter(
                    name="parent_id",
                    type="number",
                    description="The ID of the parent work item",
                    required=True,
                ),
            ],
            returns="List of child work items with their details",
            examples=[
                "get_work_item_children(parent_id=100)  # Get stories under Epic #100",
            ],
            category="context",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        """Execute get_work_item_children tool."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        parent_id = int(params["parent_id"])

        try:
            async with create_devops_client(context) as client:
                # Get parent first to show context
                parent = await client.get_work_item(
                    work_item_id=parent_id,
                    project=context.project_name,
                )

                children = await client.get_work_item_children(
                    work_item_id=parent_id,
                    project=context.project_name,
                )

                child_data = [
                    {
                        "id": child.id,
                        "type": child.work_item_type,
                        "title": child.title,
                        "state": child.state,
                        "description": child.description[:200] if child.description else None,
                        "context_string": child.to_context_string(),
                    }
                    for child in children
                ]

                return ToolResult.success(
                    data={
                        "parent": {
                            "id": parent.id,
                            "type": parent.work_item_type,
                            "title": parent.title,
                        },
                        "children": child_data,
                        "count": len(child_data),
                    },
                    message=f"Found {len(children)} children under {parent.work_item_type} #{parent_id}",
                )

        except NotFoundError:
            return ToolResult.failure(
                error=f"Work item #{parent_id} not found",
            )
        except AzureDevOpsError as e:
            logger.error(f"Failed to get work item children: {e}")
            return ToolResult.failure(error=str(e))


# =============================================================================
# Get Work Item Hierarchy Tool
# =============================================================================


class GetWorkItemHierarchyTool(BaseTool):
    """Get full hierarchy of a work item (parent and children)."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_work_item_hierarchy",
            description=(
                "Get the full hierarchy of a work item including its parent and children. "
                "Useful for understanding the complete context of a work item."
            ),
            parameters=[
                ToolParameter(
                    name="work_item_id",
                    type="number",
                    description="The ID of the work item",
                    required=True,
                ),
            ],
            returns="Work item with its parent and children",
            examples=[
                "get_work_item_hierarchy(work_item_id=150)",
            ],
            category="context",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        """Execute get_work_item_hierarchy tool."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        work_item_id = int(params["work_item_id"])

        try:
            async with create_devops_client(context) as client:
                hierarchy = await client.get_work_item_hierarchy(
                    work_item_id=work_item_id,
                    project=context.project_name,
                )

                work_item: WorkItem = hierarchy["work_item"]
                parent: WorkItem | None = hierarchy.get("parent")
                children: list[WorkItem] = hierarchy.get("children", [])

                result = {
                    "work_item": {
                        "id": work_item.id,
                        "type": work_item.work_item_type,
                        "title": work_item.title,
                        "state": work_item.state,
                        "description": work_item.description,
                        "context_string": work_item.to_context_string(),
                    },
                    "parent": None,
                    "children": [],
                }

                if parent:
                    result["parent"] = {
                        "id": parent.id,
                        "type": parent.work_item_type,
                        "title": parent.title,
                        "state": parent.state,
                    }

                if children:
                    result["children"] = [
                        {
                            "id": child.id,
                            "type": child.work_item_type,
                            "title": child.title,
                            "state": child.state,
                        }
                        for child in children
                    ]

                return ToolResult.success(
                    data=result,
                    message=f"Retrieved hierarchy for {work_item.work_item_type} #{work_item_id}",
                )

        except NotFoundError:
            return ToolResult.failure(
                error=f"Work item #{work_item_id} not found",
            )
        except AzureDevOpsError as e:
            logger.error(f"Failed to get work item hierarchy: {e}")
            return ToolResult.failure(error=str(e))


# =============================================================================
# Tool Registry
# =============================================================================

CONTEXT_TOOLS: list[BaseTool] = register_stateless_tools([
    GetWorkItemTool(),
    SearchWorkItemsTool(),
    GetWorkItemChildrenTool(),
    GetWorkItemHierarchyTool(),
])

