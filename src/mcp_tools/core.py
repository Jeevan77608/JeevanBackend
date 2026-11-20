"""
Core MCP Tools for Azure DevOps.

Provides fundamental operations:
- get_user_profile: Get authenticated user info
- list_organizations: List accessible organizations
- list_projects: List projects in an organization
- select_project: Set active project for session
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.clients.devops import AzureDevOpsClient, AzureDevOpsError

if TYPE_CHECKING:
    from src.session.manager import SessionManager
from src.config.settings import get_settings
from src.mcp_tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from src.mcp_tools.registry import register_tools

logger = logging.getLogger(__name__)


# =============================================================================
# Get User Profile Tool
# =============================================================================


class GetUserProfileTool(BaseTool):
    """Get the authenticated user's profile."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_user_profile",
            description="Get the currently authenticated user's profile information including name, email, and tenant.",
            parameters=[],
            returns="User profile with id, display_name, email, and tenant_id",
            category="core",
            examples=[
                "Get my profile information",
                "Who am I logged in as?",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute get_user_profile."""
        context.require_auth()

        # Profile is typically stored in session, return from context
        return ToolResult.success(
            data={
                "user_id": context.user_id,
                "organization_url": context.organization_url,
                "project_id": context.project_id,
                "project_name": context.project_name,
            },
            message="User profile retrieved successfully",
        )


# =============================================================================
# List Organizations Tool
# =============================================================================


class ListOrganizationsTool(BaseTool):
    """List Azure DevOps organizations accessible to the user."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_organizations",
            description="List all Azure DevOps organizations the authenticated user has access to.",
            parameters=[],
            returns="List of organizations with id, name, and url",
            category="core",
            examples=[
                "Show my organizations",
                "What Azure DevOps organizations can I access?",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute list_organizations."""
        context.require_auth()

        settings = get_settings()

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
            ) as client:
                orgs = await client.get_organizations()

            return ToolResult.success(
                data=[
                    {"id": org.id, "name": org.name, "url": org.url}
                    for org in orgs
                ],
                message=f"Found {len(orgs)} organization(s)",
            )

        except AzureDevOpsError as e:
            logger.error(f"Failed to list organizations: {e}")
            return ToolResult.failure(str(e))


# =============================================================================
# List Projects Tool
# =============================================================================


class ListProjectsTool(BaseTool):
    """List projects in an Azure DevOps organization."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_projects",
            description="List all projects in the specified or current Azure DevOps organization.",
            parameters=[
                ToolParameter(
                    name="organization_url",
                    type="string",
                    description="Organization URL (e.g., https://dev.azure.com/myorg). Uses current org if not specified.",
                    required=False,
                ),
            ],
            returns="List of projects with id, name, description, and state",
            category="core",
            examples=[
                "List projects in my organization",
                "Show all projects",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute list_projects."""
        context.require_auth()

        params = self.validate_params(**kwargs)
        org_url = params.get("organization_url") or context.organization_url

        if not org_url:
            return ToolResult.failure(
                "No organization specified. Provide organization_url or select an organization first."
            )

        settings = get_settings()
        use_pat = not settings.auth_enabled

        try:
            async with AzureDevOpsClient(
                settings=settings.azure_devops,
                access_token=context.access_token,
                organization_url=org_url,
                use_pat=use_pat,
            ) as client:
                projects = await client.get_projects()

                return ToolResult.success(
                    data=[
                        {
                            "id": p.id,
                            "name": p.name,
                            "description": p.description,
                            "state": p.state,
                            "url": p.url,
                        }
                        for p in projects
                    ],
                    message=f"Found {len(projects)} project(s)",
                )

        except AzureDevOpsError as e:
            logger.error(f"Failed to list projects: {e}")
            return ToolResult.failure(str(e))


# =============================================================================
# Select Project Tool
# =============================================================================


class SelectProjectTool(BaseTool):
    """Select a project as the active context."""

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        """Initialize with optional session manager for board context storage."""
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="select_project",
            description="Select an Azure DevOps project as the active context for subsequent operations.",
            parameters=[
                ToolParameter(
                    name="project_id",
                    type="string",
                    description="Project ID or name to select",
                    required=True,
                ),
                ToolParameter(
                    name="organization_url",
                    type="string",
                    description="Organization URL. Uses current org if not specified.",
                    required=False,
                ),
            ],
            returns="Selected project details",
            category="core",
            examples=[
                "Select project MyProject",
                "Switch to project with ID abc-123",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute select_project."""
        context.require_auth()

        params = self.validate_params(**kwargs)
        project_id = params["project_id"]
        org_url = params.get("organization_url") or context.organization_url

        if not org_url:
            return ToolResult.failure(
                "No organization specified. Provide organization_url or select an organization first."
            )

        settings = get_settings()
        use_pat = not settings.auth_enabled

        try:
            async with AzureDevOpsClient(
                settings=settings.azure_devops,
                access_token=context.access_token,
                organization_url=org_url,
                use_pat=use_pat,
            ) as client:
                project = await client.get_project(project_id)

                # Fetch and store board context automatically
                board_context = None
                if self._session_manager:
                    try:
                        logger.info(f"Fetching board context for project {project.name}")
                        board_context = await client.build_board_context(
                            project=project.name,
                            org_url=org_url,
                        )
                        await self._session_manager.store_board_context(
                            session_id=context.session_id,
                            project_id=project.name,
                            board_context=board_context,
                        )
                        logger.info(f"Stored board context: {board_context.allowed_types}")
                    except Exception as e:
                        # Don't fail if board context fetch fails - just log
                        logger.warning(f"Could not fetch board context: {e}")

            # Prepare return data
            result_data = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "state": project.state,
                "organization_url": org_url,
            }

            # Include board context info if available
            if board_context:
                result_data["board_context"] = {
                    "allowed_types": board_context.allowed_types,
                    "hierarchy": board_context.get_hierarchy_description(),
                    "default_requirement_type": board_context.default_requirement_type,
                }

            # Return project info - session update handled by caller
            return ToolResult.success(
                data=result_data,
                message=f"Selected project: {project.name}",
                update_context={
                    "project_id": project.id,
                    "project_name": project.name,
                    "organization_url": org_url,
                },
            )

        except AzureDevOpsError as e:
            logger.error(f"Failed to select project: {e}")
            return ToolResult.failure(str(e))


# =============================================================================
# Get Work Item Types Tool
# =============================================================================


class GetWorkItemTypesTool(BaseTool):
    """Get available work item types for the current project."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_work_item_types",
            description="Get the available work item types (User Story, Bug, Task, etc.) for the current project.",
            parameters=[],
            returns="List of work item types with name, reference_name, and description",
            category="core",
            examples=[
                "What work item types are available?",
                "Show me the types of items I can create",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute get_work_item_types."""
        context.require_auth()
        context.require_project()

        settings = get_settings()

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
                organization_url=context.organization_url,
            ) as client:
                types = await client.get_work_item_types(context.project_name)

            return ToolResult.success(
                data=[
                    {
                        "name": t.name,
                        "reference_name": t.reference_name,
                        "description": t.description,
                        "color": t.color,
                    }
                    for t in types
                ],
                message=f"Found {len(types)} work item type(s)",
            )

        except AzureDevOpsError as e:
            logger.error(f"Failed to get work item types: {e}")
            return ToolResult.failure(str(e))


# =============================================================================
# Tool Registry
# =============================================================================


@register_tools
def create_core_tools(session_manager: SessionManager | None = None) -> list[BaseTool]:
    """Create core tools with optional session manager dependency."""
    return [
        GetUserProfileTool(),
        ListOrganizationsTool(),
        ListProjectsTool(),
        SelectProjectTool(session_manager=session_manager),
        GetWorkItemTypesTool(),
    ]


# Default tools without session manager (for backwards compatibility)
CORE_TOOLS: list[BaseTool] = create_core_tools()

