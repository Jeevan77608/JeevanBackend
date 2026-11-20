from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.session.file_content_store import FileContentStore
from src.session.memory_store import InMemoryStore


@pytest.mark.asyncio
async def test_supabase_storage_saves_private_text_and_expiry() -> None:
    settings = SimpleNamespace(
        supabase_storage_enabled=True,
        supabase_storage_bucket="user-context-files",
    )
    store = InMemoryStore()
    supabase_client = SimpleNamespace(arequest=AsyncMock())

    with patch(
        "src.session.file_content_store.SupabaseClient",
        return_value=supabase_client,
    ):
        file_store = FileContentStore(store, timedelta(hours=1), settings)
        await file_store.store_file_content("session-123", "../requirements.txt", "text")

    storage_call, metadata_call = supabase_client.arequest.await_args_list
    assert storage_call.args[0] == "POST"
    assert storage_call.args[1].startswith(
        "/storage/v1/object/user-context-files/session-123/"
    )
    assert "../" not in storage_call.args[1]
    assert metadata_call.args[:2] == ("POST", "/rest/v1/session_files")
    metadata = metadata_call.kwargs["json_body"][0]
    assert metadata["session_id"] == "session-123"
    assert metadata["file_id"] == "../requirements.txt"
    assert metadata["storage_path"].startswith("session-123/")
    assert metadata["expires_at"]


@pytest.mark.asyncio
async def test_local_file_cleanup_handles_empty_session() -> None:
    store = InMemoryStore()
    await store.initialize()
    file_store = FileContentStore(store, timedelta(hours=1))

    await file_store.delete_session_files("missing-session")

    await store.close()


@pytest.mark.asyncio
async def test_supabase_file_retrieval_requires_unexpired_metadata() -> None:
    settings = SimpleNamespace(
        supabase_storage_enabled=True,
        supabase_storage_bucket="user-context-files",
    )
    store = InMemoryStore()
    supabase_client = SimpleNamespace(
        arequest=AsyncMock(return_value=SimpleNamespace(json=lambda: []))
    )

    with patch(
        "src.session.file_content_store.SupabaseClient",
        return_value=supabase_client,
    ):
        file_store = FileContentStore(store, timedelta(hours=1), settings)
        content = await file_store.get_file_content("session-123", "requirements.txt")

    assert content is None
    assert supabase_client.arequest.await_count == 1
    metadata_call = supabase_client.arequest.await_args
    assert metadata_call.args == ("GET", "/rest/v1/session_files")
    assert metadata_call.kwargs["params"]["expires_at"].startswith("gt.")


@pytest.mark.asyncio
async def test_supabase_file_retrieval_reads_the_private_object() -> None:
    settings = SimpleNamespace(
        supabase_storage_enabled=True,
        supabase_storage_bucket="user-context-files",
    )
    store = InMemoryStore()
    metadata_response = SimpleNamespace(
        json=lambda: [{"storage_path": "session-123/hashed-file.txt"}]
    )
    file_response = SimpleNamespace(status_code=200, content=b"file context")
    supabase_client = SimpleNamespace(
        arequest=AsyncMock(side_effect=[metadata_response, file_response])
    )

    with patch(
        "src.session.file_content_store.SupabaseClient",
        return_value=supabase_client,
    ):
        file_store = FileContentStore(store, timedelta(hours=1), settings)
        content = await file_store.get_file_content("session-123", "requirements.txt")

    assert content == "file context"
    storage_call = supabase_client.arequest.await_args_list[1]
    assert storage_call.args[0] == "GET"
    assert storage_call.args[1].startswith("/storage/v1/object/user-context-files/")
    assert storage_call.kwargs["expected_statuses"] == (404,)
