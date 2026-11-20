"""Pydantic model for normalizing frontend draft snapshots to backend shape."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.models.work_items import DraftStatus


class DraftSnapshotItem(BaseModel):
    """
    Normalizes frontend draft snapshots (camelCase, id/type) into the
    backend DraftWorkItem shape. Replaces _normalize_draft_snapshot_item().

    Handles:
    - camelCase -> snake_case field aliases
    - Type coercion (str -> datetime, str -> int, str -> float)
    - Default values for missing fields
    """

    draft_id: str = Field(..., alias="id")
    work_item_type: str = Field(..., alias="type")
    title: str
    description: str | None = Field(default=None)
    acceptance_criteria: str | None = Field(default=None, alias="acceptanceCriteria")
    area_path: str | None = Field(default=None, alias="areaPath")
    iteration_path: str | None = Field(default=None, alias="iterationPath")
    assigned_to: str | None = Field(default=None, alias="assignedTo")
    parent_draft_id: str | None = Field(default=None, alias="parentDraftId")
    story_points: float | None = Field(default=None, alias="storyPoints")
    priority: int | None = Field(default=None)
    status: DraftStatus = Field(default=DraftStatus.DRAFT)
    tags: list[str] = Field(default_factory=list)
    custom_fields: dict[str, Any] = Field(default_factory=dict, alias="customFields")
    links: list[Any] = Field(default_factory=list)
    validation_errors: list[str] = Field(
        default_factory=list, alias="validationErrors"
    )
    azure_devops_id: int | None = Field(default=None, alias="pushedId")
    azure_devops_url: str | None = Field(default=None, alias="azureDevopsUrl")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), alias="createdAt"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), alias="updatedAt"
    )

    model_config = {"populate_by_name": True}

    @field_validator("priority", mode="before")
    @classmethod
    def coerce_priority(cls, v: Any) -> int | None:
        if v is None:
            return None
        try:
            return max(1, min(4, int(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("story_points", mode="before")
    @classmethod
    def coerce_story_points(cls, v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v: Any) -> DraftStatus:
        if v is None:
            return DraftStatus.DRAFT
        if isinstance(v, str):
            try:
                return DraftStatus(v.lower())
            except ValueError:
                return DraftStatus.DRAFT
        return v

    @field_validator("azure_devops_id", mode="before")
    @classmethod
    def coerce_azure_devops_id(cls, v: Any) -> int | None:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def coerce_datetime(cls, v: Any) -> datetime:
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return datetime.now(timezone.utc)
        return datetime.now(timezone.utc)
