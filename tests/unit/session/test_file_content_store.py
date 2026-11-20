"""Tests for src/session/file_content_store.py — uploaded file content storage."""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.session.file_content_store import FileContentStore
from src.session.memory_store import InMemoryStore

SESSION_ID = "sess-file-test-001"


@pytest.fixture()
async def store():
    mem = InMemoryStore()
    await mem.initialize()
    file_store = FileContentStore(mem, timedelta(hours=1))
    yield file_store
    await mem.close()


class TestFileContentStore:
    """Verify FileContentStore operations for uploaded file content."""

    async def test_store_and_get_file_content(self, store: FileContentStore) -> None:
        await store.store_file_content(SESSION_ID, "file-001", "Hello, world!")
        result = await store.get_file_content(SESSION_ID, "file-001")
        assert result == "Hello, world!"

    async def test_get_file_content_returns_none_when_missing(self, store: FileContentStore) -> None:
        result = await store.get_file_content(SESSION_ID, "nonexistent-file")
        assert result is None

    async def test_list_file_contents_returns_file_ids(self, store: FileContentStore) -> None:
        await store.store_file_content(SESSION_ID, "file-a", "Content A")
        await store.store_file_content(SESSION_ID, "file-b", "Content B")
        await store.store_file_content(SESSION_ID, "file-c", "Content C")

        file_ids = await store.list_file_contents(SESSION_ID)
        assert sorted(file_ids) == ["file-a", "file-b", "file-c"]

    async def test_list_file_contents_empty_when_no_files(self, store: FileContentStore) -> None:
        file_ids = await store.list_file_contents("empty-session")
        assert file_ids == []

    async def test_store_multiple_files_different_ids(self, store: FileContentStore) -> None:
        await store.store_file_content(SESSION_ID, "readme", "# README")
        await store.store_file_content(SESSION_ID, "notes", "Some notes")

        readme = await store.get_file_content(SESSION_ID, "readme")
        notes = await store.get_file_content(SESSION_ID, "notes")
        assert readme == "# README"
        assert notes == "Some notes"

    async def test_overwrite_file_content(self, store: FileContentStore) -> None:
        await store.store_file_content(SESSION_ID, "file-001", "Original")
        await store.store_file_content(SESSION_ID, "file-001", "Updated")

        result = await store.get_file_content(SESSION_ID, "file-001")
        assert result == "Updated"
