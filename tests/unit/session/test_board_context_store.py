"""Tests for src/session/board_context_store.py — project board context caching."""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.models.devops import BoardContext, ProcessTemplate
from src.session.board_context_store import BoardContextStore
from src.session.memory_store import InMemoryStore

SESSION_ID = "sess-board-test-001"
PROJECT_ID = "proj-001"


def _make_board_context(
    project_id: str = PROJECT_ID,
    process_template: ProcessTemplate | None = ProcessTemplate.AGILE,
) -> BoardContext:
    return BoardContext(
        project_id=project_id,
        process_template=process_template,
        allowed_types=["Epic", "User Story", "Task", "Bug"],
        hierarchy_levels=[
            {"name": "Task", "rank": 1},
            {"name": "User Story", "rank": 2},
            {"name": "Epic", "rank": 3},
        ],
        default_requirement_type="User Story",
        parent_types_for_tasks=["User Story"],
        epic_type="Epic",
        task_type="Task",
        bug_type="Bug",
        feature_type="Feature",
        story_points_field="story_points",
    )


@pytest.fixture()
async def store():
    mem = InMemoryStore()
    await mem.initialize()
    board_store = BoardContextStore(mem, timedelta(hours=1))
    yield board_store
    await mem.close()


class TestBoardContextStore:
    """Verify BoardContextStore caching and retrieval operations."""

    async def test_store_and_get_board_context(self, store: BoardContextStore) -> None:
        ctx = _make_board_context()
        await store.store_board_context(SESSION_ID, PROJECT_ID, ctx)

        result = await store.get_board_context(SESSION_ID, PROJECT_ID)
        assert result is not None
        assert result.project_id == PROJECT_ID
        assert result.process_template == ProcessTemplate.AGILE
        assert "Epic" in result.allowed_types
        assert "User Story" in result.allowed_types
        assert result.default_requirement_type == "User Story"
        assert result.task_type == "Task"
        assert result.bug_type == "Bug"
        assert result.feature_type == "Feature"

    async def test_get_board_context_returns_none_when_missing(self, store: BoardContextStore) -> None:
        result = await store.get_board_context("no-session", "no-project")
        assert result is None

    async def test_store_board_context_overwrites(self, store: BoardContextStore) -> None:
        original = _make_board_context()
        await store.store_board_context(SESSION_ID, PROJECT_ID, original)

        updated = _make_board_context()
        updated.default_requirement_type = "Product Backlog Item"
        updated.process_template = ProcessTemplate.SCRUM
        await store.store_board_context(SESSION_ID, PROJECT_ID, updated)

        result = await store.get_board_context(SESSION_ID, PROJECT_ID)
        assert result is not None
        assert result.default_requirement_type == "Product Backlog Item"
        assert result.process_template == ProcessTemplate.SCRUM

    async def test_different_projects_stored_separately(self, store: BoardContextStore) -> None:
        ctx_a = _make_board_context(project_id="proj-a")
        ctx_b = _make_board_context(project_id="proj-b")
        ctx_b.default_requirement_type = "Issue"
        ctx_b.process_template = ProcessTemplate.BASIC

        await store.store_board_context(SESSION_ID, "proj-a", ctx_a)
        await store.store_board_context(SESSION_ID, "proj-b", ctx_b)

        result_a = await store.get_board_context(SESSION_ID, "proj-a")
        result_b = await store.get_board_context(SESSION_ID, "proj-b")

        assert result_a is not None
        assert result_b is not None
        assert result_a.project_id == "proj-a"
        assert result_b.project_id == "proj-b"
        assert result_a.default_requirement_type == "User Story"
        assert result_b.default_requirement_type == "Issue"
        assert result_a.process_template == ProcessTemplate.AGILE
        assert result_b.process_template == ProcessTemplate.BASIC
