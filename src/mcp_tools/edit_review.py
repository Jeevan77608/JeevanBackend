"""
Edit and Review MCP Tools for Azure DevOps.

This module provides tools for managing draft work items before pushing
to Azure DevOps. Drafts are stored in the user's session and can be
reviewed, edited, and validated before final submission.

Tools:
    - list_drafts: List all draft work items in session
    - get_draft: Get a specific draft work item by ID
    - update_draft: Update draft fields (title, description, etc.)
    - delete_draft: Remove a draft from the session
    - preview_items: Preview formatted work items before push
    - validate_items: Validate drafts against Azure DevOps rules

Draft Lifecycle:
    1. Created by generation tools (generate_user_stories, etc.)
    2. Stored in session with unique draft_id
    3. Can be refined using refine_item tool
    4. Validated using validate_items
    5. Pushed to Azure DevOps using push tools
    6. Removed from session after successful push

Validation Rules:
    - Required fields (title, work item type)
    - Field length limits
    - Valid state values
    - Parent-child relationship constraints

Example:
    # List all drafts
    list_drafts()

    # Update a draft's title
    update_draft(draft_id="abc123", updates={"title": "New Title"})

    # Validate before pushing
    validate_items()

    # Delete unwanted draft
    delete_draft(draft_id="abc123")
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
from src.mcp_tools.conversations import (
    restore_session_drafts_from_conversation,
    sync_drafts_to_conversation,
)
from src.mcp_tools.registry import register_tools
from src.session.manager import SessionManager

logger = logging.getLogger(__name__)


# =============================================================================
# List Drafts Tool
# =============================================================================


class ListDraftsTool(BaseTool):
    """List all draft work items in the session."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_drafts",
            description="List all draft work items in the current session.",
            parameters=[
                ToolParameter(
                    name="work_item_type",
                    type="string",
                    description="Filter by work item type (User Story, Task, Bug, etc.)",
                    required=False,
                ),
            ],
            returns="List of draft work items",
            category="edit",
            examples=[
                "Show me all my drafts",
                "List draft user stories",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute list_drafts."""
        params = self.validate_params(**kwargs)
        work_item_type = params.get("work_item_type")

        # If session has no drafts but conversation has drafts_snapshot, restore so UI and push see epics
        await restore_session_drafts_from_conversation(context, self._session_manager)

        draft_collection = await self._session_manager.get_drafts(
            session_id=context.session_id,
        )

        drafts = draft_collection.items if draft_collection else []

        if work_item_type:
            drafts = [d for d in drafts if d.work_item_type == work_item_type]

        return ToolResult.success(
            data={
                "items": [
                    {
                        "draft_id": d.draft_id,
                        "work_item_type": d.work_item_type,
                        "title": d.title,
                        "description": d.description,
                        "acceptance_criteria": d.acceptance_criteria,
                        "priority": d.priority,
                        "story_points": d.story_points,
                        "parent_draft_id": d.parent_draft_id,
                        "created_at": d.created_at.isoformat(),
                    }
                    for d in drafts
                ]
            },
            message=f"Found {len(drafts)} draft(s)",
        )


# =============================================================================
# Get Draft Tool
# =============================================================================


class GetDraftTool(BaseTool):
    """Get a specific draft work item."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_draft",
            description="Get the details of a specific draft work item.",
            parameters=[
                ToolParameter(
                    name="draft_id",
                    type="string",
                    description="ID of the draft work item",
                    required=True,
                ),
            ],
            returns="Draft work item details",
            category="edit",
            examples=[
                "Show me draft abc-123",
                "Get details of the first draft",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute get_draft."""
        params = self.validate_params(**kwargs)
        draft_id = params["draft_id"]

        draft = await self._session_manager.get_draft(
            session_id=context.session_id,
            draft_id=draft_id,
        )

        if not draft:
            return ToolResult.failure(f"Draft not found: {draft_id}")

        return ToolResult.success(
            data={
                "draft_id": draft.draft_id,
                "type": draft.work_item_type,
                "title": draft.title,
                "description": draft.description,
                "acceptance_criteria": draft.acceptance_criteria,
                "priority": draft.priority,
                "story_points": draft.story_points,
                "created_at": draft.created_at.isoformat(),
            },
            message=f"Draft: {draft.title}",
        )


# =============================================================================
# Update Draft Tool
# =============================================================================


class UpdateDraftTool(BaseTool):
    """Update a draft work item."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="update_draft",
            description="Update fields of a draft work item.",
            parameters=[
                ToolParameter(
                    name="draft_id",
                    type="string",
                    description="ID of the draft work item",
                    required=True,
                ),
                ToolParameter(
                    name="title",
                    type="string",
                    description="New title",
                    required=False,
                ),
                ToolParameter(
                    name="description",
                    type="string",
                    description="New description",
                    required=False,
                ),
                ToolParameter(
                    name="work_item_type",
                    type="string",
                    description="Work item type (Epic, Feature, User Story, Task, Bug, Issue)",
                    required=False,
                ),
                ToolParameter(
                    name="fields",
                    type="object",
                    description="Additional fields to update",
                    required=False,
                ),
            ],
            returns="Updated draft work item",
            category="edit",
            examples=[
                "Update the title of draft abc-123",
                "Change the type of a draft to Epic",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute update_draft."""
        params = self.validate_params(**kwargs)
        draft_id = params["draft_id"]

        draft = await self._session_manager.get_draft(
            session_id=context.session_id,
            draft_id=draft_id,
        )

        # If draft not in session but we have a saved conversation, restore from its drafts_snapshot
        # so editing an epic (e.g. after load) finds the draft and save succeeds
        if not draft and context.conversation_id:
            await restore_session_drafts_from_conversation(context, self._session_manager)
            draft = await self._session_manager.get_draft(
                session_id=context.session_id,
                draft_id=draft_id,
            )

        if not draft:
            return ToolResult.failure(f"Draft not found: {draft_id}")

        # Update draft fields
        if params.get("title"):
            draft.title = params["title"]
        if params.get("description"):
            draft.description = params["description"]
        if params.get("acceptance_criteria"):
            draft.acceptance_criteria = params["acceptance_criteria"]
        if params.get("priority"):
            draft.priority = params["priority"]
        if params.get("story_points"):
            draft.story_points = params["story_points"]
        if params.get("work_item_type"):
            draft.work_item_type = params["work_item_type"]
        if params.get("fields"):
            # Update custom fields
            draft.custom_fields.update(params["fields"])

        # Save updated draft
        await self._session_manager.update_draft_item(
            session_id=context.session_id,
            draft_item=draft,
        )

        await sync_drafts_to_conversation(context, self._session_manager)
        return ToolResult.success(
            data={
                "draft_id": draft.draft_id,
                "type": draft.work_item_type,
                "title": draft.title,
                "description": draft.description,
                "acceptance_criteria": draft.acceptance_criteria,
                "priority": draft.priority,
                "story_points": draft.story_points,
            },
            message=f"Updated draft: {draft.title}",
        )


# =============================================================================
# Delete Draft Tool
# =============================================================================


class DeleteDraftTool(BaseTool):
    """Delete a draft work item."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="delete_draft",
            description="Delete a draft work item from the session.",
            parameters=[
                ToolParameter(
                    name="draft_id",
                    type="string",
                    description="ID of the draft work item to delete",
                    required=True,
                ),
            ],
            returns="Confirmation of deletion",
            category="edit",
            examples=[
                "Delete draft abc-123",
                "Remove the first story",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute delete_draft."""
        params = self.validate_params(**kwargs)
        draft_id = params["draft_id"]

        deleted = await self._session_manager.delete_draft(
            session_id=context.session_id,
            draft_id=draft_id,
        )

        if not deleted:
            return ToolResult.failure(f"Draft not found: {draft_id}")

        await sync_drafts_to_conversation(context, self._session_manager)
        return ToolResult.success(
            message=f"Deleted draft: {draft_id}",
        )


# =============================================================================
# Preview Items Tool
# =============================================================================


class PreviewItemsTool(BaseTool):
    """Preview work items before pushing to Azure DevOps."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="preview_items",
            description="Preview all draft work items before pushing to Azure DevOps.",
            parameters=[
                ToolParameter(
                    name="draft_ids",
                    type="array",
                    description="Specific draft IDs to preview. If not provided, previews all drafts.",
                    required=False,
                ),
            ],
            returns="Preview of work items to be created",
            category="review",
            examples=[
                "Preview all items before pushing",
                "Show me what will be created",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute preview_items."""
        context.require_project()

        params = self.validate_params(**kwargs)
        draft_ids = params.get("draft_ids")

        draft_collection = await self._session_manager.get_drafts(
            session_id=context.session_id,
        )

        drafts = draft_collection.items if draft_collection else []

        if draft_ids:
            drafts = [d for d in drafts if d.draft_id in draft_ids]

        if not drafts:
            return ToolResult.failure("No drafts to preview")

        preview = []
        for draft in drafts:
            preview.append({
                "draft_id": draft.draft_id,
                "type": draft.work_item_type,
                "title": draft.title,
                "description": draft.description,
                "acceptance_criteria": draft.acceptance_criteria,
                "priority": draft.priority,
                "story_points": draft.story_points,
                "target_project": context.project_name,
            })

        return ToolResult.success(
            data={
                "items": preview,
                "count": len(preview),
                "project": context.project_name,
            },
            message=f"Preview: {len(preview)} item(s) ready to push to {context.project_name}",
        )


# =============================================================================
# Validate Items Tool
# =============================================================================


class ValidateItemsTool(BaseTool):
    """Validate work items against project rules."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="validate_items",
            description="Validate draft work items against Azure DevOps project rules and required fields.",
            parameters=[
                ToolParameter(
                    name="draft_ids",
                    type="array",
                    description="Specific draft IDs to validate. If not provided, validates all drafts.",
                    required=False,
                ),
            ],
            returns="Validation results with any errors or warnings",
            category="review",
            examples=[
                "Validate all items before pushing",
                "Check if items are ready to create",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute validate_items."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        draft_ids = params.get("draft_ids")

        draft_collection = await self._session_manager.get_drafts(
            session_id=context.session_id,
        )

        drafts = draft_collection.items if draft_collection else []

        if draft_ids:
            drafts = [d for d in drafts if d.draft_id in draft_ids]

        if not drafts:
            return ToolResult.failure("No drafts to validate")

        settings = get_settings()
        validation_results = []

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
                organization_url=context.organization_url,
            ) as client:
                # Get work item types for validation
                types = await client.get_work_item_types(context.project_name)
                type_names = {t.name for t in types}

                for draft in drafts:
                    errors = []
                    warnings = []

                    # Check work item type exists
                    if draft.work_item_type not in type_names:
                        errors.append(f"Invalid work item type: {draft.work_item_type}")

                    # Check required fields
                    if not draft.title or not draft.title.strip():
                        errors.append("Title is required")

                    # Check title length
                    if len(draft.title) > 255:
                        errors.append("Title exceeds 255 characters")

                    validation_results.append({
                        "draft_id": draft.draft_id,
                        "title": draft.title,
                        "valid": len(errors) == 0,
                        "errors": errors,
                        "warnings": warnings,
                    })

        except AzureDevOpsError as e:
            logger.error(f"Validation failed: {e}")
            return ToolResult.failure(f"Validation failed: {e}")

        all_valid = all(r["valid"] for r in validation_results)

        return ToolResult.success(
            data={
                "results": validation_results,
                "all_valid": all_valid,
                "total": len(validation_results),
                "valid_count": sum(1 for r in validation_results if r["valid"]),
            },
            message="All items valid" if all_valid else "Some items have validation errors",
        )


# =============================================================================
# Tool Factory
# =============================================================================


@register_tools
def create_edit_review_tools(session_manager: SessionManager) -> list[BaseTool]:
    """Create edit/review tools with session manager dependency."""
    return [
        ListDraftsTool(session_manager),
        GetDraftTool(session_manager),
        UpdateDraftTool(session_manager),
        DeleteDraftTool(session_manager),
        PreviewItemsTool(session_manager),
        ValidateItemsTool(session_manager),
    ]
