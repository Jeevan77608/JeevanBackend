"""
Push MCP Tools for Azure DevOps.

This module provides tools for pushing draft work items to Azure DevOps.
It is the final step in the generation workflow:

    Context → Generate → Review → Push

Tools:
    - push_to_devops: Push all or selected drafts to Azure DevOps
    - push_single: Push a single draft work item

Features:
    - Batch creation with transaction-like behavior
    - Automatic parent-child linking
    - Area and iteration path assignment
    - Validation before push
    - Detailed success/failure reporting

Workflow:
    1. Generate work items (stored as drafts in session)
    2. Review and refine drafts using edit tools
    3. Validate drafts using validate_items
    4. Push to Azure DevOps using push_to_devops

Example:
    # Push all drafts with area/iteration paths
    push_to_devops(
        area_path="MyProject\\Team A",
        iteration_path="MyProject\\Sprint 1"
    )

    # Push specific drafts only
    push_to_devops(
        draft_ids=["abc123", "def456"],
        area_path="MyProject\\Team A"
    )

Error Handling:
    - Returns partial success if some items fail
    - Failed items remain as drafts for retry
    - Successful items are removed from drafts
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
    ensure_session_drafts_from_conversation,
    restore_session_drafts_from_conversation,
    sync_drafts_to_conversation,
)
from src.mcp_tools.registry import register_tools
from src.session.manager import SessionManager

logger = logging.getLogger(__name__)


# =============================================================================
# Work item type resolution (process template compatibility)
# =============================================================================
# Draft types (e.g. "User Story") are mapped to project types. Basic process
# uses "Issue" instead of "User Story"; Scrum uses "Product Backlog Item".
# Epic has fallbacks so Basic process (no Epic type) still creates parent items as Issue/Feature.
_WIT_TYPE_ALTERNATIVES: dict[str, list[str]] = {
    "User Story": ["User Story", "Issue", "Product Backlog Item", "Requirement"],
    "Issue": ["Issue", "User Story", "Product Backlog Item"],
    "Task": ["Task"],
    "Epic": ["Epic", "Feature", "Issue"],  # Basic has no Epic: create as Feature or Issue so epics push to board
    "Feature": ["Feature", "Epic", "Issue"],
    "Bug": ["Bug"],
}


def _resolve_work_item_type(
    draft_type: str,
    project_type_names: set[str],
) -> str:
    """Resolve draft work item type to a type that exists in the project."""
    draft_type = (draft_type or "").strip() or "User Story"
    alternatives = _WIT_TYPE_ALTERNATIVES.get(draft_type, [draft_type])
    for candidate in alternatives:
        if candidate in project_type_names:
            return candidate
    # Fallback: use draft type if project has it, else first alternative
    return alternatives[0]


# =============================================================================
# Push to DevOps Tool
# =============================================================================


class PushToDevOpsTool(BaseTool):
    """Push draft work items to Azure DevOps."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="push_to_devops",
            description="Create work items in Azure DevOps from draft work items.",
            parameters=[
                ToolParameter(
                    name="draft_ids",
                    type="array",
                    description="Specific draft IDs to push. If not provided, pushes all drafts.",
                    required=False,
                ),
                ToolParameter(
                    name="area_path",
                    type="string",
                    description="Area path for the work items",
                    required=False,
                ),
                ToolParameter(
                    name="iteration_path",
                    type="string",
                    description="Iteration path for the work items",
                    required=False,
                ),
                ToolParameter(
                    name="additional_tags",
                    type="array",
                    description="Additional tags to apply to all work items (merged with default tag and draft tags, deduplicated)",
                    required=False,
                ),
            ],
            returns="Created work items with their Azure DevOps IDs",
            category="push",
            examples=[
                "Push all drafts to Azure DevOps",
                "Create the work items in DevOps",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute push_to_devops."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        draft_ids = params.get("draft_ids")
        area_path = params.get("area_path")
        iteration_path = params.get("iteration_path")
        additional_tags: list[str] = params.get("additional_tags") or []

        # When user has a conversation open, use its drafts_snapshot so we push what the UI shows
        # (e.g. 1 epic + 6 tasks). Otherwise session might have only tasks and epic would be skipped.
        if context.conversation_id:
            replaced = await ensure_session_drafts_from_conversation(context, self._session_manager)
            if replaced:
                logger.info(
                    "push_to_devops: loaded drafts from conversation %s before push",
                    context.conversation_id[:8],
                )

        draft_collection = await self._session_manager.get_drafts(
            session_id=context.session_id,
        )

        drafts = draft_collection.items if draft_collection else []

        # If session still empty (e.g. no snapshot in conversation), try restore for backward compat
        if not drafts and context.conversation_id:
            await restore_session_drafts_from_conversation(context, self._session_manager)
            draft_collection = await self._session_manager.get_drafts(session_id=context.session_id)
            drafts = draft_collection.items if draft_collection else []

        if draft_ids:
            drafts = [d for d in drafts if d.draft_id in draft_ids]

        if not drafts:
            return ToolResult.failure("No drafts to push")

        # Build draft_id set for lookup
        draft_ids = {d.draft_id for d in drafts}

        # Sort drafts: push parents first (items without parent, or whose parent is not in batch),
        # then children. This ensures we can resolve parent_draft_id -> work_item_id when creating.
        def push_order(d: Any) -> tuple[int, str]:
            has_parent_in_batch = (
                d.parent_draft_id is not None and d.parent_draft_id in draft_ids
            )
            return (0 if not has_parent_in_batch else 1, d.draft_id)

        drafts_sorted = sorted(drafts, key=push_order)

        settings = get_settings()
        created_items = []
        failed_items = []
        draft_id_to_work_item_id: dict[str, int] = {}
        type_mapping: dict[str, str] = {}  # draft_type -> actual created type (e.g. Epic -> Issue)

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
                organization_url=context.organization_url,
            ) as client:
                # Resolve work item types to project types (e.g. User Story -> Issue for Basic process)
                project_types = await client.get_work_item_types(context.project_name)
                project_type_names = {t.name for t in project_types}

                for draft in drafts_sorted:
                    try:
                        resolved_type = _resolve_work_item_type(
                            draft.work_item_type, project_type_names
                        )
                        if (draft.work_item_type or "").strip() and resolved_type != (draft.work_item_type or "").strip():
                            type_mapping[draft.work_item_type] = resolved_type

                        # Build fields from draft properties
                        fields: dict[str, Any] = {
                            "System.Title": draft.title,
                        }
                        if draft.description:
                            fields["System.Description"] = draft.description
                        if draft.acceptance_criteria:
                            fields["Microsoft.VSTS.Common.AcceptanceCriteria"] = draft.acceptance_criteria
                        if draft.priority:
                            fields["Microsoft.VSTS.Common.Priority"] = draft.priority
                        # Effort/Story Points: Basic process uses Effort; Agile uses Story Points. Set the right one.
                        if draft.story_points is not None:
                            if resolved_type == "Issue":
                                fields["Microsoft.VSTS.Scheduling.Effort"] = draft.story_points
                            else:
                                fields["Microsoft.VSTS.Scheduling.StoryPoints"] = draft.story_points
                                # Epic/Feature in some processes use Effort
                                if resolved_type in ("Epic", "Feature"):
                                    fields["Microsoft.VSTS.Scheduling.Effort"] = draft.story_points
                        if draft.assigned_to:
                            fields["System.AssignedTo"] = draft.assigned_to
                        # Inject default push tag + additional_tags (deduped)
                        effective_tags = list(draft.tags) if draft.tags else []
                        if settings.default_push_tag and settings.default_push_tag not in effective_tags:
                            effective_tags.insert(0, settings.default_push_tag)
                        for tag in additional_tags:
                            tag = tag.strip()
                            if tag and tag not in effective_tags:
                                effective_tags.append(tag)
                        if effective_tags:
                            fields["System.Tags"] = "; ".join(effective_tags)

                        # Add area/iteration paths if specified
                        if area_path:
                            fields["System.AreaPath"] = area_path
                        elif draft.area_path:
                            fields["System.AreaPath"] = draft.area_path
                        if iteration_path:
                            fields["System.IterationPath"] = iteration_path
                        elif draft.iteration_path:
                            fields["System.IterationPath"] = draft.iteration_path

                        # Add custom fields
                        fields.update(draft.custom_fields)

                        # Resolve parent: if this draft has parent_draft_id and parent was already pushed, link to it
                        parent_ado_id: int | None = None
                        if draft.parent_draft_id:
                            parent_ado_id = draft_id_to_work_item_id.get(draft.parent_draft_id)

                        # Create work item (without parent link - add separately for reliability)
                        work_item = await client.create_work_item(
                            project=context.project_name,
                            work_item_type=resolved_type,
                            fields=fields,
                            parent_id=None,  # Add link after creation
                        )

                        work_item_id = work_item.get("id")
                        if work_item_id is None:
                            # API returned but no id - treat as failure so we don't count or delete draft
                            failed_items.append({
                                "draft_id": draft.draft_id,
                                "title": draft.title,
                                "type": draft.work_item_type,
                                "error": "Azure DevOps API did not return a work item id",
                            })
                            continue

                        draft_id_to_work_item_id[draft.draft_id] = work_item_id

                        # Add parent-child link if we have a parent (Epic/User Story -> Task)
                        if parent_ado_id is not None:
                            try:
                                await client.add_work_item_link(
                                    project=context.project_name,
                                    work_item_id=work_item_id,
                                    target_id=parent_ado_id,
                                    link_type="System.LinkTypes.Hierarchy-Reverse",
                                )
                                logger.info(
                                    "Linked work item %s to parent %s",
                                    work_item_id,
                                    parent_ado_id,
                                )
                            except AzureDevOpsError as link_err:
                                logger.warning(
                                    "Failed to link work item %s to parent %s: %s",
                                    work_item_id,
                                    parent_ado_id,
                                    link_err,
                                )

                        work_item_type = work_item.get("fields", {}).get("System.WorkItemType", draft.work_item_type)
                        created_items.append({
                            "draft_id": draft.draft_id,
                            "work_item_id": work_item_id,
                            "type": work_item_type,
                            "title": work_item.get("fields", {}).get("System.Title", draft.title),
                            "url": work_item.get("url"),
                        })

                        # Remove draft from session only after confirmed creation
                        await self._session_manager.delete_draft(
                            session_id=context.session_id,
                            draft_id=draft.draft_id,
                        )

                    except AzureDevOpsError as e:
                        logger.error("Failed to create work item: %s", e)
                        failed_items.append({
                            "draft_id": draft.draft_id,
                            "title": draft.title,
                            "type": draft.work_item_type,
                            "error": str(e),
                        })

        except AzureDevOpsError as e:
            logger.error("Push failed: %s", e)
            return ToolResult.failure(f"Push failed: {e}")

        if failed_items and not created_items:
            return ToolResult.failure(
                f"All {len(failed_items)} item(s) failed to create",
            )

        # Build summary by type so the UI/LLM can show "Epics: 2, Tasks: 6" and explain failures
        created_by_type: dict[str, int] = {}
        for item in created_items:
            t = item.get("type") or "Work Item"
            created_by_type[t] = created_by_type.get(t, 0) + 1
        failed_by_type: dict[str, list[str]] = {}
        for item in failed_items:
            t = item.get("type") or "Work Item"
            if t not in failed_by_type:
                failed_by_type[t] = []
            failed_by_type[t].append(item.get("error") or "Unknown error")

        summary_parts = []
        if created_by_type:
            summary_parts.append(", ".join(f"{count} {t}" for t, count in sorted(created_by_type.items())))
        if failed_by_type:
            for t, errs in failed_by_type.items():
                summary_parts.append(f"{len(errs)} {t} failed (e.g. {errs[0][:80]})")
        summary_message = "Created: " + "; ".join(summary_parts) if summary_parts else "No items created."

        await sync_drafts_to_conversation(context, self._session_manager)
        data: dict[str, Any] = {
            "created": created_items,
            "failed": failed_items,
            "created_by_type": created_by_type,
            "failed_by_type": {k: v for k, v in failed_by_type.items()},
            "summary": summary_message,
        }
        if type_mapping:
            data["type_mapping"] = type_mapping
        return ToolResult.success(data=data, message=summary_message)


# =============================================================================
# Push Single Tool
# =============================================================================


class PushSingleTool(BaseTool):
    """Push a single draft work item to Azure DevOps."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="push_single",
            description="Create a single work item in Azure DevOps from a draft.",
            parameters=[
                ToolParameter(
                    name="draft_id",
                    type="string",
                    description="ID of the draft work item to push",
                    required=True,
                ),
                ToolParameter(
                    name="area_path",
                    type="string",
                    description="Area path for the work item",
                    required=False,
                ),
                ToolParameter(
                    name="iteration_path",
                    type="string",
                    description="Iteration path for the work item",
                    required=False,
                ),
                ToolParameter(
                    name="parent_id",
                    type="number",
                    description="Parent work item ID to link to",
                    required=False,
                ),
            ],
            returns="Created work item with Azure DevOps ID",
            category="push",
            examples=[
                "Push draft abc-123 to Azure DevOps",
                "Create this story in DevOps",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute push_single."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        draft_id = params["draft_id"]
        area_path = params.get("area_path")
        iteration_path = params.get("iteration_path")
        parent_id = params.get("parent_id")

        draft = await self._session_manager.get_draft(
            session_id=context.session_id,
            draft_id=draft_id,
        )

        if not draft:
            return ToolResult.failure(f"Draft not found: {draft_id}")

        settings = get_settings()

        try:
            async with AzureDevOpsClient(
                settings.azure_devops,
                context.access_token,
                organization_url=context.organization_url,
            ) as client:
                # Resolve work item type to project type (e.g. User Story -> Issue for Basic process)
                project_types = await client.get_work_item_types(context.project_name)
                project_type_names = {t.name for t in project_types}
                resolved_type = _resolve_work_item_type(
                    draft.work_item_type, project_type_names
                )

                # Build fields from draft properties
                fields: dict[str, Any] = {
                    "System.Title": draft.title,
                }
                if draft.description:
                    fields["System.Description"] = draft.description
                if draft.acceptance_criteria:
                    fields["Microsoft.VSTS.Common.AcceptanceCriteria"] = draft.acceptance_criteria
                if draft.priority:
                    fields["Microsoft.VSTS.Common.Priority"] = draft.priority
                # Effort/Story Points: Basic process uses Effort; Agile uses Story Points.
                if draft.story_points is not None:
                    if resolved_type == "Issue":
                        fields["Microsoft.VSTS.Scheduling.Effort"] = draft.story_points
                    else:
                        fields["Microsoft.VSTS.Scheduling.StoryPoints"] = draft.story_points
                        if resolved_type in ("Epic", "Feature"):
                            fields["Microsoft.VSTS.Scheduling.Effort"] = draft.story_points
                if draft.assigned_to:
                    fields["System.AssignedTo"] = draft.assigned_to
                # Inject default push tag (deduped)
                effective_tags = list(draft.tags) if draft.tags else []
                if settings.default_push_tag and settings.default_push_tag not in effective_tags:
                    effective_tags.insert(0, settings.default_push_tag)
                if effective_tags:
                    fields["System.Tags"] = "; ".join(effective_tags)

                # Add area/iteration paths
                if area_path:
                    fields["System.AreaPath"] = area_path
                elif draft.area_path:
                    fields["System.AreaPath"] = draft.area_path
                if iteration_path:
                    fields["System.IterationPath"] = iteration_path
                elif draft.iteration_path:
                    fields["System.IterationPath"] = draft.iteration_path

                # Add custom fields
                fields.update(draft.custom_fields)

                # Create work item using resolved type (e.g. Issue instead of User Story)
                work_item = await client.create_work_item(
                    project=context.project_name,
                    work_item_type=resolved_type,
                    fields=fields,
                )

                work_item_id = work_item.get("id")
                work_item_title = work_item.get("fields", {}).get("System.Title", draft.title)
                work_item_url = work_item.get("url")
                work_item_type = work_item.get("fields", {}).get("System.WorkItemType", draft.work_item_type)

                # Add parent link if specified
                if parent_id and work_item_id:
                    await client.add_work_item_link(
                        project=context.project_name,
                        work_item_id=work_item_id,
                        target_id=parent_id,
                        link_type="System.LinkTypes.Hierarchy-Reverse",
                    )

                # Remove draft from session
                await self._session_manager.delete_draft(
                    session_id=context.session_id,
                    draft_id=draft_id,
                )

                await sync_drafts_to_conversation(context, self._session_manager)
                return ToolResult.success(
                    data={
                        "work_item_id": work_item_id,
                        "title": work_item_title,
                        "url": work_item_url,
                        "type": work_item_type,
                    },
                    message=f"Created work item #{work_item_id}: {work_item_title}",
                )

        except AzureDevOpsError as e:
            logger.error(f"Failed to create work item: {e}")
            return ToolResult.failure(f"Failed to create work item: {e}")


# =============================================================================
# Tool Factory
# =============================================================================


@register_tools
def create_push_tools(session_manager: SessionManager) -> list[BaseTool]:
    """Create push tools with session manager dependency."""
    return [
        PushToDevOpsTool(session_manager),
        PushSingleTool(session_manager),
    ]
