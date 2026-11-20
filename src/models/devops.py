"""
Azure DevOps related models and API response validation.

This module provides Pydantic models for:
- Azure DevOps entities (Organizations, Projects, Work Items, etc.)
- API response validation with strict type checking
- Work item hierarchy and relationships
"""

from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ProcessTemplate(str, Enum):
    """Azure DevOps process templates."""

    AGILE = "Agile"
    SCRUM = "Scrum"
    CMMI = "CMMI"
    BASIC = "Basic"


class BoardContext(BaseModel):
    """
    Board context for AI generation - optimized structure with work item types and hierarchy.
    Used to make AI prompts dynamic based on the actual project board structure.
    """

    project_id: str = Field(..., description="Project ID")
    process_template: ProcessTemplate | None = Field(
        default=None, description="Process template (Agile, Scrum, CMMI, Basic)"
    )

    # All available work item types
    allowed_types: list[str] = Field(
        default_factory=list,
        description="All work item type names available in the project (e.g., Epic, User Story, Task, Bug, Issue, Product Backlog Item)",
    )

    # Hierarchy information
    hierarchy_levels: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Backlog levels with their types in rank order (from Epic down to Task)",
    )

    # Default types by category
    default_requirement_type: str = Field(
        default="Epic",
        description="Default backlog requirement type (User Story, Issue, Product Backlog Item, etc.)",
    )

    parent_types_for_tasks: list[str] = Field(
        default_factory=lambda: ["User Story"],
        description="Work item types that can be parents of Tasks",
    )

    epic_type: str | None = Field(
        default="Epic",
        description="Epic/portfolio level type name (Epic, etc.)",
    )

    task_type: str = Field(
        default="Task",
        description="Task type name (Task, etc.)",
    )

    bug_type: str = Field(
        default="Bug",
        description="Bug type name (Bug, etc.)",
    )

    feature_type: str | None = Field(
        default=None,
        description="Feature type name if it exists (Feature, etc.)",
    )

    # Field hints for prompts
    story_points_field: str = Field(
        default="story_points",
        description="Field name for effort estimation (story_points, effort, etc.)",
    )

    def get_hierarchy_description(self) -> str:
        """Generate a human-readable hierarchy description for AI prompts."""
        if not self.hierarchy_levels:
            return f"{self.epic_type or 'Epic'} → {self.default_requirement_type} → {self.task_type}"

        # Build from hierarchy levels
        level_names = [level.get("name", "") for level in self.hierarchy_levels if level.get("name")]
        if level_names:
            return " → ".join(reversed(level_names))  # Reverse because rank is bottom-up

        return f"{self.epic_type or 'Epic'} → {self.default_requirement_type} → {self.task_type}"

    def get_allowed_types_string(self) -> str:
        """Get comma-separated list of allowed types for prompts."""
        return ", ".join(self.allowed_types) if self.allowed_types else "User Story, Epic, Task, Bug"


class Organization(BaseModel):
    """Azure DevOps organization."""

    id: str = Field(..., description="Organization ID")
    name: str = Field(..., description="Organization name")
    url: str = Field(..., description="Organization URL")


class Project(BaseModel):
    """Azure DevOps project."""

    id: str = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    description: str | None = Field(default=None, description="Project description")
    url: str = Field(..., description="Project URL")
    state: str = Field(..., description="Project state")
    process_template: ProcessTemplate | None = Field(
        default=None, description="Process template type"
    )


class WorkItemType(BaseModel):
    """Azure DevOps work item type."""

    name: str = Field(..., description="Work item type name")
    reference_name: str = Field(..., description="Reference name")
    description: str | None = Field(default=None)
    color: str | None = Field(default=None)
    icon: str | None = Field(default=None)
    is_disabled: bool = Field(default=False)


class WorkItemField(BaseModel):
    """Azure DevOps work item field."""

    name: str = Field(..., description="Field name")
    reference_name: str = Field(..., description="Field reference name")
    field_type: str = Field(..., description="Field type")
    is_required: bool = Field(default=False)
    allowed_values: list[str] = Field(default_factory=list)


class Board(BaseModel):
    """A board in Azure DevOps."""

    id: str = Field(..., description="Board ID")
    name: str = Field(..., description="Board name")
    url: str = Field(default="", description="Board URL")


class BoardStructure(BaseModel):
    """Project board structure."""

    project_id: str = Field(..., description="Project ID")
    process_template: ProcessTemplate = Field(..., description="Process template")
    work_item_types: list[WorkItemType] = Field(default_factory=list)
    hierarchy: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Work item hierarchy (parent -> children types)",
    )
    custom_fields: dict[str, list[WorkItemField]] = Field(
        default_factory=dict,
        description="Custom fields per work item type",
    )


class AreaPath(BaseModel):
    """Area path classification node."""

    id: int = Field(..., description="Node ID")
    name: str = Field(..., description="Node name")
    path: str = Field(..., description="Full path")
    has_children: bool = Field(default=False)
    children: list["AreaPath"] = Field(default_factory=list)


class IterationPath(BaseModel):
    """Iteration path classification node."""

    id: int = Field(..., description="Node ID")
    name: str = Field(..., description="Node name")
    path: str = Field(..., description="Full path")
    start_date: datetime | None = Field(default=None)
    finish_date: datetime | None = Field(default=None)
    has_children: bool = Field(default=False)
    children: list["IterationPath"] = Field(default_factory=list)


class Wiki(BaseModel):
    """Azure DevOps wiki."""

    id: str = Field(..., description="Wiki ID")
    name: str = Field(..., description="Wiki name")
    project_id: str = Field(..., description="Project ID")
    type: str = Field(..., description="Wiki type (projectWiki or codeWiki)")
    url: str = Field(..., description="Wiki URL")


class WikiPage(BaseModel):
    """Azure DevOps wiki page."""

    id: str | int = Field(default="", description="Page ID")
    path: str = Field(..., description="Page path")
    content: str | None = Field(default=None, description="Page content in markdown")
    git_item_path: str | None = Field(default=None, alias="gitItemPath")
    order: int | None = Field(default=None, description="Page order")
    sub_pages: list["WikiPage"] = Field(default_factory=list, alias="subPages")
    url: str | None = Field(default=None, description="Remote URL to the wiki page")


# =============================================================================
# Work Item Models (for context extraction)
# =============================================================================


class WorkItemLink(BaseModel):
    """Work item relationship link."""

    rel: str = Field(..., description="Relationship type (e.g., System.LinkTypes.Hierarchy-Forward)")
    url: str = Field(..., description="URL to the related work item")

    @property
    def target_id(self) -> int | None:
        """Extract work item ID from URL."""
        try:
            # URL format: https://dev.azure.com/org/_apis/wit/workItems/123
            return int(self.url.split("/")[-1])
        except (ValueError, IndexError):
            return None


class WorkItemRelation(BaseModel):
    """Work item relation details."""

    rel: str = Field(..., description="Relationship type")
    url: str = Field(..., description="Target URL")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Relation attributes")


class WorkItem(BaseModel):
    """
    Azure DevOps work item with full field support.

    Used for reading existing work items for context.
    """

    id: int = Field(..., description="Work item ID")
    rev: int = Field(default=1, description="Revision number")
    url: str = Field(default="", description="Work item URL")
    fields: dict[str, Any] = Field(default_factory=dict, description="Work item fields")
    relations: list[WorkItemRelation] = Field(default_factory=list, description="Work item relations")

    @property
    def work_item_type(self) -> str:
        """Get work item type from fields."""
        return self.fields.get("System.WorkItemType", "Unknown")

    @property
    def title(self) -> str:
        """Get work item title from fields."""
        return self.fields.get("System.Title", "Untitled")

    @property
    def description(self) -> str | None:
        """Get work item description from fields."""
        return self.fields.get("System.Description")

    @property
    def state(self) -> str:
        """Get work item state from fields."""
        return self.fields.get("System.State", "Unknown")

    @property
    def area_path(self) -> str:
        """Get area path from fields."""
        return self.fields.get("System.AreaPath", "")

    @property
    def iteration_path(self) -> str:
        """Get iteration path from fields."""
        return self.fields.get("System.IterationPath", "")

    @property
    def assigned_to(self) -> str | None:
        """Get assigned to from fields."""
        assigned = self.fields.get("System.AssignedTo")
        if isinstance(assigned, dict):
            return assigned.get("displayName")
        return assigned

    @property
    def parent_id(self) -> int | None:
        """Get parent work item ID from relations."""
        for relation in self.relations:
            if "Hierarchy-Reverse" in relation.rel:
                try:
                    return int(relation.url.split("/")[-1])
                except (ValueError, IndexError):
                    pass
        return None

    @property
    def child_ids(self) -> list[int]:
        """Get child work item IDs from relations."""
        children = []
        for relation in self.relations:
            if "Hierarchy-Forward" in relation.rel:
                try:
                    children.append(int(relation.url.split("/")[-1]))
                except (ValueError, IndexError):
                    pass
        return children

    def to_context_string(self) -> str:
        """Convert work item to a context string for AI generation."""
        lines = [
            f"[{self.work_item_type}] #{self.id}: {self.title}",
            f"State: {self.state}",
        ]
        if self.description:
            # Truncate very long descriptions but keep enough detail for context
            desc = self.description[:2000] + "..." if len(self.description) > 2000 else self.description
            lines.append(f"Description: {desc}")
        if self.area_path:
            lines.append(f"Area: {self.area_path}")
        if self.iteration_path:
            lines.append(f"Iteration: {self.iteration_path}")

        # Include acceptance criteria if present
        acceptance_criteria = self.fields.get("Microsoft.VSTS.Common.AcceptanceCriteria")
        if acceptance_criteria:
            ac = acceptance_criteria[:1000] + "..." if len(acceptance_criteria) > 1000 else acceptance_criteria
            lines.append(f"Acceptance Criteria: {ac}")

        # Include story points if present
        story_points = self.fields.get("Microsoft.VSTS.Scheduling.StoryPoints")
        if story_points is not None:
            lines.append(f"Story Points: {story_points}")

        return "\n".join(lines)


# =============================================================================
# API Response Models (for validation)
# =============================================================================


class ApiListResponse(BaseModel, Generic[T]):
    """Generic Azure DevOps API list response."""

    count: int = Field(default=0, description="Number of items")
    value: list[T] = Field(default_factory=list, description="List of items")


class OrganizationApiResponse(BaseModel):
    """Organization API response from VSSPS."""

    accountId: str = Field(..., alias="accountId")
    accountName: str = Field(..., alias="accountName")
    accountUri: str | None = Field(default=None, alias="accountUri")


class ProjectApiResponse(BaseModel):
    """Project API response."""

    id: str
    name: str
    description: str | None = None
    url: str | None = None
    state: str = "wellFormed"
    visibility: str | None = None
    lastUpdateTime: datetime | None = None


class WorkItemApiResponse(BaseModel):
    """Work item API response for validation."""

    model_config = {"populate_by_name": True}

    id: int
    rev: int = 1
    url: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    links: dict[str, Any] = Field(default_factory=dict, alias="_links")

    def to_work_item(self) -> WorkItem:
        """Convert API response to WorkItem model."""
        return WorkItem(
            id=self.id,
            rev=self.rev,
            url=self.url,
            fields=self.fields,
            relations=[
                WorkItemRelation(
                    rel=r.get("rel", ""),
                    url=r.get("url", ""),
                    attributes=r.get("attributes", {}),
                )
                for r in self.relations
            ],
        )


class WorkItemQueryResult(BaseModel):
    """WIQL query result."""

    queryType: str = "flat"
    queryResultType: str = "workItem"
    asOf: datetime | None = None
    workItems: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def work_item_ids(self) -> list[int]:
        """Extract work item IDs from query result."""
        return [wi.get("id") for wi in self.workItems if "id" in wi]


class WorkItemBatchResponse(BaseModel):
    """Batch work items response."""

    count: int = 0
    value: list[WorkItemApiResponse] = Field(default_factory=list)

    def to_work_items(self) -> list[WorkItem]:
        """Convert to list of WorkItem models."""
        return [wi.to_work_item() for wi in self.value]
