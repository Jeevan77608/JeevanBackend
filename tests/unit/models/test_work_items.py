"""Tests for src/models/work_items.py — DraftStatus, DraftWorkItem, WorkItemLink,
DraftCollection, ValidationResult, PushResult."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.work_items import (
    DraftCollection,
    DraftStatus,
    DraftWorkItem,
    PushResult,
    ValidationResult,
    WorkItemLink,
)


# =============================================================================
# DraftStatus
# =============================================================================


class TestDraftStatus:
    """Tests for the DraftStatus enum."""

    def test_draft_value(self) -> None:
        assert DraftStatus.DRAFT == "draft"

    def test_validated_value(self) -> None:
        assert DraftStatus.VALIDATED == "validated"

    def test_pushed_value(self) -> None:
        assert DraftStatus.PUSHED == "pushed"

    def test_error_value(self) -> None:
        assert DraftStatus.ERROR == "error"

    def test_all_members_present(self) -> None:
        names = {s.name for s in DraftStatus}
        assert names == {"DRAFT", "VALIDATED", "PUSHED", "ERROR"}


# =============================================================================
# DraftWorkItem
# =============================================================================


class TestDraftWorkItem:
    """Tests for the DraftWorkItem model."""

    def test_creation_with_minimal_fields(self) -> None:
        item = DraftWorkItem(
            draft_id="d-1",
            work_item_type="Task",
            title="Do something",
        )
        assert item.draft_id == "d-1"
        assert item.work_item_type == "Task"
        assert item.title == "Do something"

    def test_default_status_is_draft(self) -> None:
        item = DraftWorkItem(
            draft_id="d-2",
            work_item_type="Bug",
            title="Fix it",
        )
        assert item.status == DraftStatus.DRAFT

    def test_default_optional_fields(self) -> None:
        item = DraftWorkItem(
            draft_id="d-3",
            work_item_type="User Story",
            title="Story",
        )
        assert item.description is None
        assert item.acceptance_criteria is None
        assert item.area_path is None
        assert item.iteration_path is None
        assert item.assigned_to is None
        assert item.priority is None
        assert item.story_points is None
        assert item.tags == []
        assert item.custom_fields == {}
        assert item.links == []
        assert item.parent_draft_id is None
        assert item.validation_errors == []
        assert item.azure_devops_id is None
        assert item.azure_devops_url is None

    def test_creation_with_all_fields(self) -> None:
        item = DraftWorkItem(
            draft_id="d-full",
            work_item_type="Epic",
            title="Epic Title",
            description="<p>desc</p>",
            acceptance_criteria="AC",
            area_path="Project\\Area",
            iteration_path="Project\\Sprint1",
            assigned_to="dev@co.com",
            priority=2,
            story_points=8.0,
            tags=["mvp", "backend"],
            status=DraftStatus.VALIDATED,
            parent_draft_id="d-parent",
        )
        assert item.priority == 2
        assert item.story_points == 8.0
        assert item.tags == ["mvp", "backend"]
        assert item.status == DraftStatus.VALIDATED
        assert item.parent_draft_id == "d-parent"

    def test_priority_minimum_bound(self) -> None:
        item = DraftWorkItem(
            draft_id="d-p1",
            work_item_type="Task",
            title="T",
            priority=1,
        )
        assert item.priority == 1

    def test_priority_maximum_bound(self) -> None:
        item = DraftWorkItem(
            draft_id="d-p4",
            work_item_type="Task",
            title="T",
            priority=4,
        )
        assert item.priority == 4

    def test_priority_below_minimum_raises(self) -> None:
        with pytest.raises(ValidationError):
            DraftWorkItem(
                draft_id="d-bad",
                work_item_type="Task",
                title="T",
                priority=0,
            )

    def test_priority_above_maximum_raises(self) -> None:
        with pytest.raises(ValidationError):
            DraftWorkItem(
                draft_id="d-bad",
                work_item_type="Task",
                title="T",
                priority=5,
            )

    def test_tags_list_preserved(self) -> None:
        item = DraftWorkItem(
            draft_id="d-tags",
            work_item_type="Task",
            title="T",
            tags=["a", "b", "c"],
        )
        assert item.tags == ["a", "b", "c"]


# =============================================================================
# WorkItemLink
# =============================================================================


class TestWorkItemLink:
    """Tests for the WorkItemLink model."""

    def test_creation_with_target_id(self) -> None:
        link = WorkItemLink(
            rel="System.LinkTypes.Hierarchy-Forward",
            target_id="42",
        )
        assert link.rel == "System.LinkTypes.Hierarchy-Forward"
        assert link.target_id == "42"
        assert link.target_draft_id is None

    def test_creation_with_target_draft_id(self) -> None:
        link = WorkItemLink(
            rel="System.LinkTypes.Hierarchy-Forward",
            target_draft_id="d-child",
        )
        assert link.target_id is None
        assert link.target_draft_id == "d-child"


# =============================================================================
# DraftCollection
# =============================================================================


class TestDraftCollection:
    """Tests for the DraftCollection model."""

    def test_get_item_found(self, sample_draft_collection: DraftCollection) -> None:
        found = sample_draft_collection.get_item("draft-001")
        assert found is not None
        assert found.draft_id == "draft-001"

    def test_get_item_not_found(self, sample_draft_collection: DraftCollection) -> None:
        assert sample_draft_collection.get_item("nonexistent") is None

    def test_get_hierarchy_items_without_parents(self) -> None:
        item_a = DraftWorkItem(draft_id="a", work_item_type="Epic", title="A")
        item_b = DraftWorkItem(draft_id="b", work_item_type="Epic", title="B")
        collection = DraftCollection(
            session_id="s",
            project_id="p",
            items=[item_a, item_b],
        )
        hierarchy = collection.get_hierarchy()
        # Both have parent_draft_id=None, so they are under the None key
        assert None in hierarchy
        assert len(hierarchy[None]) == 2

    def test_get_hierarchy_items_with_parents(self) -> None:
        parent = DraftWorkItem(draft_id="parent", work_item_type="Epic", title="Parent")
        child1 = DraftWorkItem(draft_id="c1", work_item_type="Story", title="C1", parent_draft_id="parent")
        child2 = DraftWorkItem(draft_id="c2", work_item_type="Story", title="C2", parent_draft_id="parent")
        collection = DraftCollection(
            session_id="s",
            project_id="p",
            items=[parent, child1, child2],
        )
        hierarchy = collection.get_hierarchy()
        assert len(hierarchy[None]) == 1  # parent
        assert len(hierarchy["parent"]) == 2  # child1, child2

    def test_get_hierarchy_empty_collection(self) -> None:
        collection = DraftCollection(session_id="s", project_id="p", items=[])
        assert collection.get_hierarchy() == {}


# =============================================================================
# ValidationResult
# =============================================================================


class TestValidationResult:
    """Tests for the ValidationResult model."""

    def test_creation_valid(self) -> None:
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True

    def test_default_errors_and_warnings_empty(self) -> None:
        result = ValidationResult(is_valid=True)
        assert result.errors == {}
        assert result.warnings == {}

    def test_creation_with_errors(self) -> None:
        result = ValidationResult(
            is_valid=False,
            errors={"d-1": ["Title missing"]},
            warnings={"d-1": ["Priority not set"]},
        )
        assert result.is_valid is False
        assert result.errors["d-1"] == ["Title missing"]
        assert result.warnings["d-1"] == ["Priority not set"]


# =============================================================================
# PushResult
# =============================================================================


class TestPushResult:
    """Tests for the PushResult model."""

    def test_creation_success(self) -> None:
        result = PushResult(success=True)
        assert result.success is True

    def test_default_lists_empty(self) -> None:
        result = PushResult(success=True)
        assert result.pushed_items == []
        assert result.failed_items == []
        assert result.errors == []

    def test_creation_with_errors(self) -> None:
        result = PushResult(
            success=False,
            errors=["API timeout", "Auth failed"],
        )
        assert result.success is False
        assert len(result.errors) == 2
