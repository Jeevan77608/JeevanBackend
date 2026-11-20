"""Tests for src/session/draft_repository.py — draft work item CRUD operations."""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.models.work_items import DraftCollection, DraftStatus, DraftWorkItem
from src.session.draft_repository import DraftRepository
from src.session.exceptions import SessionNotFoundError
from src.session.memory_store import InMemoryStore

SESSION_ID = "sess-draft-test-001"
PROJECT_ID = "TestProject"


def _make_draft(draft_id: str = "draft-001", title: str = "Test Story") -> DraftWorkItem:
    return DraftWorkItem(
        draft_id=draft_id,
        work_item_type="User Story",
        title=title,
        description="A test draft",
        priority=2,
        story_points=5.0,
        status=DraftStatus.DRAFT,
    )


def _make_collection(
    *items: DraftWorkItem,
    session_id: str = SESSION_ID,
    project_id: str = PROJECT_ID,
) -> DraftCollection:
    return DraftCollection(
        session_id=session_id,
        project_id=project_id,
        items=list(items),
    )


@pytest.fixture()
async def repo():
    mem = InMemoryStore()
    await mem.initialize()
    draft_repo = DraftRepository(mem, timedelta(hours=1))
    yield draft_repo
    await mem.close()


class TestDraftRepository:
    """Verify DraftRepository CRUD operations using InMemoryStore."""

    async def test_save_and_get_drafts_round_trip(self, repo: DraftRepository) -> None:
        draft = _make_draft()
        collection = _make_collection(draft)
        await repo.save_drafts(SESSION_ID, collection)

        result = await repo.get_drafts(SESSION_ID)
        assert result is not None
        assert len(result.items) == 1
        assert result.items[0].draft_id == "draft-001"
        assert result.items[0].title == "Test Story"
        assert result.session_id == SESSION_ID
        assert result.project_id == PROJECT_ID

    async def test_get_drafts_returns_none_when_empty(self, repo: DraftRepository) -> None:
        result = await repo.get_drafts("nonexistent-session")
        assert result is None

    async def test_delete_drafts_removes_data(self, repo: DraftRepository) -> None:
        collection = _make_collection(_make_draft())
        await repo.save_drafts(SESSION_ID, collection)

        await repo.delete_drafts(SESSION_ID)
        result = await repo.get_drafts(SESSION_ID)
        assert result is None

    async def test_delete_drafts_no_error_when_missing(self, repo: DraftRepository) -> None:
        # Should not raise any exception
        await repo.delete_drafts("never-existed")

    async def test_replace_drafts_from_snapshot_camelcase(self, repo: DraftRepository) -> None:
        snapshot = [
            {
                "id": "d-1",
                "type": "User Story",
                "title": "T",
                "acceptanceCriteria": "AC",
            },
        ]
        await repo.replace_drafts_from_snapshot(SESSION_ID, snapshot, PROJECT_ID)

        result = await repo.get_drafts(SESSION_ID)
        assert result is not None
        assert len(result.items) == 1
        assert result.items[0].draft_id == "d-1"
        assert result.items[0].work_item_type == "User Story"
        assert result.items[0].title == "T"
        assert result.items[0].acceptance_criteria == "AC"

    async def test_replace_drafts_from_snapshot_skips_invalid(self, repo: DraftRepository) -> None:
        snapshot = [
            {"id": "d-good", "type": "Task", "title": "Valid"},
            {"bad": "data"},  # missing required fields
            "not-a-dict",  # not even a dict
        ]
        await repo.replace_drafts_from_snapshot(SESSION_ID, snapshot, PROJECT_ID)

        result = await repo.get_drafts(SESSION_ID)
        assert result is not None
        assert len(result.items) == 1
        assert result.items[0].draft_id == "d-good"

    async def test_update_draft_item_replaces_existing(self, repo: DraftRepository) -> None:
        original = _make_draft("draft-001", "Original Title")
        collection = _make_collection(original)
        await repo.save_drafts(SESSION_ID, collection)

        updated = _make_draft("draft-001", "Updated Title")
        result = await repo.update_draft_item(SESSION_ID, updated)
        assert len(result.items) == 1
        assert result.items[0].title == "Updated Title"

    async def test_update_draft_item_appends_new(self, repo: DraftRepository) -> None:
        existing = _make_draft("draft-001", "Existing")
        collection = _make_collection(existing)
        await repo.save_drafts(SESSION_ID, collection)

        new_item = _make_draft("draft-002", "Brand New")
        result = await repo.update_draft_item(SESSION_ID, new_item)
        assert len(result.items) == 2
        ids = [item.draft_id for item in result.items]
        assert "draft-001" in ids
        assert "draft-002" in ids

    async def test_update_draft_item_raises_when_no_drafts(self, repo: DraftRepository) -> None:
        new_item = _make_draft("draft-999", "Orphan")
        with pytest.raises(SessionNotFoundError):
            await repo.update_draft_item("no-session", new_item)

    async def test_get_draft_found(self, repo: DraftRepository) -> None:
        collection = _make_collection(
            _make_draft("draft-001", "First"),
            _make_draft("draft-002", "Second"),
        )
        await repo.save_drafts(SESSION_ID, collection)

        result = await repo.get_draft(SESSION_ID, "draft-002")
        assert result is not None
        assert result.title == "Second"

    async def test_get_draft_not_found(self, repo: DraftRepository) -> None:
        collection = _make_collection(_make_draft("draft-001", "Only"))
        await repo.save_drafts(SESSION_ID, collection)

        result = await repo.get_draft(SESSION_ID, "draft-nonexistent")
        assert result is None

    async def test_delete_draft_removes_item(self, repo: DraftRepository) -> None:
        collection = _make_collection(
            _make_draft("draft-001", "Keep"),
            _make_draft("draft-002", "Remove"),
        )
        await repo.save_drafts(SESSION_ID, collection)

        removed = await repo.delete_draft(SESSION_ID, "draft-002")
        assert removed is True

        result = await repo.get_drafts(SESSION_ID)
        assert result is not None
        assert len(result.items) == 1
        assert result.items[0].draft_id == "draft-001"

    async def test_delete_draft_returns_false_when_missing(self, repo: DraftRepository) -> None:
        collection = _make_collection(_make_draft("draft-001", "Only"))
        await repo.save_drafts(SESSION_ID, collection)

        removed = await repo.delete_draft(SESSION_ID, "draft-ghost")
        assert removed is False
