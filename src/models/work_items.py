"""Work item models for draft and final states."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DraftStatus(str, Enum):
    """Draft work item status."""

    DRAFT = "draft"
    VALIDATED = "validated"
    PUSHED = "pushed"
    ERROR = "error"


class WorkItemLink(BaseModel):
    """Work item link/relation."""

    rel: str = Field(..., description="Relation type (e.g., 'System.LinkTypes.Hierarchy-Forward')")
    target_id: str | None = Field(default=None, description="Target work item ID (for existing items)")
    target_draft_id: str | None = Field(default=None, description="Target draft ID (for draft items)")


class DraftWorkItem(BaseModel):
    """Draft work item (not yet pushed to Azure DevOps)."""

    draft_id: str = Field(..., description="Unique draft identifier")
    work_item_type: str = Field(..., description="Work item type (Epic, Story, Task, etc.)")
    title: str = Field(..., description="Work item title")
    description: str | None = Field(default=None, description="Work item description (HTML)")
    acceptance_criteria: str | None = Field(default=None, description="Acceptance criteria")
    area_path: str | None = Field(default=None, description="Area path")
    iteration_path: str | None = Field(default=None, description="Iteration path")
    assigned_to: str | None = Field(default=None, description="Assigned to user email")
    priority: int | None = Field(default=None, ge=1, le=4, description="Priority (1-4)")
    story_points: float | None = Field(default=None, description="Story points estimate")
    tags: list[str] = Field(default_factory=list, description="Tags")
    custom_fields: dict[str, Any] = Field(default_factory=dict, description="Custom field values")
    links: list[WorkItemLink] = Field(default_factory=list, description="Work item links")
    parent_draft_id: str | None = Field(default=None, description="Parent draft ID for hierarchy")
    status: DraftStatus = Field(default=DraftStatus.DRAFT, description="Draft status")
    validation_errors: list[str] = Field(default_factory=list, description="Validation errors")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # After push
    azure_devops_id: int | None = Field(default=None, description="Azure DevOps work item ID after push")
    azure_devops_url: str | None = Field(default=None, description="Azure DevOps work item URL after push")


class DraftCollection(BaseModel):
    """Collection of draft work items for a session."""

    session_id: str = Field(..., description="Session ID")
    project_id: str = Field(..., description="Target project ID")
    items: list[DraftWorkItem] = Field(default_factory=list, description="Draft work items")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_item(self, draft_id: str) -> DraftWorkItem | None:
        """Get a draft item by ID."""
        return next((item for item in self.items if item.draft_id == draft_id), None)

    def get_hierarchy(self) -> dict[str | None, list[DraftWorkItem]]:
        """Get items organized by parent."""
        hierarchy: dict[str | None, list[DraftWorkItem]] = {}
        for item in self.items:
            parent = item.parent_draft_id
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(item)
        return hierarchy


class ValidationResult(BaseModel):
    """Validation result for draft work items."""

    is_valid: bool = Field(..., description="Whether all items are valid")
    errors: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Errors per draft item ID",
    )
    warnings: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Warnings per draft item ID",
    )


class PushResult(BaseModel):
    """Result of pushing work items to Azure DevOps."""

    success: bool = Field(..., description="Whether the push was successful")
    pushed_items: list[DraftWorkItem] = Field(default_factory=list, description="Successfully pushed items")
    failed_items: list[DraftWorkItem] = Field(default_factory=list, description="Failed items")
    errors: list[str] = Field(default_factory=list, description="Error messages")

