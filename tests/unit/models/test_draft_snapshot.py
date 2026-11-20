"""Tests for DraftSnapshotItem: field aliases, coercion, validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.draft_snapshot import DraftSnapshotItem
from src.models.work_items import DraftStatus


# =============================================================================
# Helpers
# =============================================================================


def _minimal_camelcase(**overrides: object) -> dict[str, object]:
    """Build a minimal valid camelCase payload (frontend shape).

    Args:
        **overrides: Fields to override in the payload.

    Returns:
        Dict with the minimum required fields plus any overrides.
    """
    base: dict[str, object] = {
        "id": "draft-001",
        "type": "User Story",
        "title": "Test Draft",
    }
    base.update(overrides)
    return base


def _minimal_snakecase(**overrides: object) -> dict[str, object]:
    """Build a minimal valid snake_case payload (backend shape).

    Args:
        **overrides: Fields to override in the payload.

    Returns:
        Dict with the minimum required snake_case fields plus any overrides.
    """
    base: dict[str, object] = {
        "draft_id": "draft-001",
        "work_item_type": "User Story",
        "title": "Test Draft",
    }
    base.update(overrides)
    return base


# =============================================================================
# Alias / Construction Tests
# =============================================================================


class TestDraftSnapshotConstruction:
    """Tests for creating DraftSnapshotItem via aliases and field names."""

    def test_create_from_frontend_camelcase(self) -> None:
        """Create with camelCase aliases (id, type, acceptanceCriteria, etc.)."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(
                acceptanceCriteria="Given/When/Then",
                areaPath="Project\\Team",
                iterationPath="Sprint 1",
                assignedTo="dev@example.com",
                parentDraftId="parent-001",
                storyPoints=8,
                customFields={"custom.field": "value"},
                validationErrors=["Missing description"],
                pushedId=42,
                azureDevopsUrl="https://dev.azure.com/org/proj/_workitems/edit/42",
            )
        )

        assert item.draft_id == "draft-001"
        assert item.work_item_type == "User Story"
        assert item.title == "Test Draft"
        assert item.acceptance_criteria == "Given/When/Then"
        assert item.area_path == "Project\\Team"
        assert item.iteration_path == "Sprint 1"
        assert item.assigned_to == "dev@example.com"
        assert item.parent_draft_id == "parent-001"
        assert item.story_points == 8.0
        assert item.custom_fields == {"custom.field": "value"}
        assert item.validation_errors == ["Missing description"]
        assert item.azure_devops_id == 42
        assert item.azure_devops_url == "https://dev.azure.com/org/proj/_workitems/edit/42"

    def test_create_from_snake_case(self) -> None:
        """Create with snake_case field names (populate_by_name=True)."""
        item = DraftSnapshotItem.model_validate(_minimal_snakecase())

        assert item.draft_id == "draft-001"
        assert item.work_item_type == "User Story"
        assert item.title == "Test Draft"

    def test_defaults_for_optional_fields(self) -> None:
        """Optional fields default to None or empty collections."""
        item = DraftSnapshotItem.model_validate(_minimal_camelcase())

        assert item.description is None
        assert item.acceptance_criteria is None
        assert item.area_path is None
        assert item.tags == []
        assert item.custom_fields == {}
        assert item.links == []
        assert item.validation_errors == []
        assert item.azure_devops_id is None
        assert item.azure_devops_url is None


# =============================================================================
# Priority Coercion Tests
# =============================================================================


class TestPriorityCoercion:
    """Tests for the priority field validator."""

    def test_priority_coercion_clamps_high(self) -> None:
        """Priority > 4 is clamped to 4."""
        item = DraftSnapshotItem.model_validate(_minimal_camelcase(priority=10))
        assert item.priority == 4

    def test_priority_coercion_clamps_low(self) -> None:
        """Priority < 1 is clamped to 1."""
        item = DraftSnapshotItem.model_validate(_minimal_camelcase(priority=0))
        assert item.priority == 1

    def test_priority_coercion_none(self) -> None:
        """None priority stays None."""
        item = DraftSnapshotItem.model_validate(_minimal_camelcase(priority=None))
        assert item.priority is None

    def test_priority_coercion_invalid(self) -> None:
        """Non-numeric priority becomes None."""
        item = DraftSnapshotItem.model_validate(_minimal_camelcase(priority="not-a-number"))
        assert item.priority is None

    def test_priority_coercion_string_number(self) -> None:
        """String '2' is coerced to int 2."""
        item = DraftSnapshotItem.model_validate(_minimal_camelcase(priority="2"))
        assert item.priority == 2


# =============================================================================
# Story Points Coercion Tests
# =============================================================================


class TestStoryPointsCoercion:
    """Tests for the story_points field validator."""

    def test_story_points_coercion(self) -> None:
        """String '5.5' becomes float 5.5."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(storyPoints="5.5")
        )
        assert item.story_points == 5.5

    def test_story_points_coercion_invalid(self) -> None:
        """Non-numeric becomes None."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(storyPoints="not-a-number")
        )
        assert item.story_points is None

    def test_story_points_coercion_none(self) -> None:
        """None stays None."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(storyPoints=None)
        )
        assert item.story_points is None


# =============================================================================
# Status Coercion Tests
# =============================================================================


class TestStatusCoercion:
    """Tests for the status field validator."""

    def test_status_coercion_from_string(self) -> None:
        """'draft' string converts to DraftStatus.DRAFT."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(status="draft")
        )
        assert item.status == DraftStatus.DRAFT

    def test_status_coercion_from_uppercase(self) -> None:
        """'VALIDATED' (uppercase) converts to DraftStatus.VALIDATED."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(status="VALIDATED")
        )
        assert item.status == DraftStatus.VALIDATED

    def test_status_coercion_none(self) -> None:
        """None defaults to DraftStatus.DRAFT."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(status=None)
        )
        assert item.status == DraftStatus.DRAFT

    def test_status_coercion_invalid_string(self) -> None:
        """Invalid status string defaults to DraftStatus.DRAFT."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(status="unknown-status")
        )
        assert item.status == DraftStatus.DRAFT


# =============================================================================
# Azure DevOps ID Coercion Tests
# =============================================================================


class TestAzureDevopsIdCoercion:
    """Tests for the azure_devops_id field validator."""

    def test_azure_devops_id_coercion(self) -> None:
        """String '42' becomes int 42."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(pushedId="42")
        )
        assert item.azure_devops_id == 42

    def test_azure_devops_id_coercion_none(self) -> None:
        """None stays None."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(pushedId=None)
        )
        assert item.azure_devops_id is None

    def test_azure_devops_id_coercion_invalid(self) -> None:
        """Non-numeric becomes None."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(pushedId="not-an-id")
        )
        assert item.azure_devops_id is None


# =============================================================================
# Datetime Coercion Tests
# =============================================================================


class TestDatetimeCoercion:
    """Tests for created_at / updated_at field validators."""

    def test_datetime_coercion_z_suffix(self) -> None:
        """'2025-01-01T00:00:00Z' parses correctly (Python 3.10 compat)."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(createdAt="2025-01-01T00:00:00Z")
        )
        assert item.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)

    def test_datetime_coercion_with_offset(self) -> None:
        """ISO string with explicit offset parses correctly."""
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(updatedAt="2025-06-15T12:30:00+00:00")
        )
        assert item.updated_at == datetime(2025, 6, 15, 12, 30, tzinfo=timezone.utc)

    def test_datetime_coercion_none(self) -> None:
        """None defaults to approximately now (UTC)."""
        before = datetime.now(timezone.utc)
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(createdAt=None)
        )
        after = datetime.now(timezone.utc)

        assert before <= item.created_at <= after
        assert item.created_at.tzinfo is not None

    def test_datetime_coercion_naive_datetime(self) -> None:
        """Naive datetime gets UTC timezone attached."""
        naive = datetime(2025, 3, 15, 10, 0, 0)
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(createdAt=naive)
        )
        assert item.created_at.tzinfo == timezone.utc
        assert item.created_at.year == 2025

    def test_datetime_coercion_invalid_string(self) -> None:
        """Invalid datetime string defaults to now."""
        before = datetime.now(timezone.utc)
        item = DraftSnapshotItem.model_validate(
            _minimal_camelcase(createdAt="not-a-date")
        )
        after = datetime.now(timezone.utc)

        assert before <= item.created_at <= after
