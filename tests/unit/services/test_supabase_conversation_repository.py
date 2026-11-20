from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.config.settings import clear_settings_cache
from src.services import conversation_repository as repository


@pytest.fixture()
def supabase_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "APP_SECRET_KEY": "test-secret-key-must-be-32-chars!",
        "AUTH_ENABLED": "true",
        "AUTH_PROVIDER": "supabase",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_ANON_KEY": "test-anon-key",
        "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
        "CHAT_SAVE_ENABLED": "true",
        "APP_ENV": "development",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-api-key",
    }.items():
        monkeypatch.setenv(key, value)
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_supabase_save_uses_owner_scoped_upsert(
    monkeypatch: pytest.MonkeyPatch,
    supabase_environment: None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    saved_row = {
        "id": "conversation-1",
        "user_id": "4b36d83b-d58b-4db4-b215-5529ed4b2b2c",
        "title": "Requirements",
        "markdown": "",
        "messages_snapshot": [{"role": "user", "content": "Create a Jira task"}],
        "context_snapshot": None,
        "drafts_snapshot": None,
        "created_at": now,
        "updated_at": now,
    }
    request = MagicMock()
    request.return_value = request
    request.json.return_value = [saved_row]
    request.is_success = True
    monkeypatch.setattr(repository, "_supabase_request", request)

    conversation = repository.save_conversation(
        saved_row["user_id"],
        conversation_id="conversation-1",
        title="Requirements",
        messages_snapshot=saved_row["messages_snapshot"],
    )

    assert conversation.id == "conversation-1"
    assert conversation.user_id == saved_row["user_id"]
    call_args = request.call_args
    assert call_args.args == ("POST",)
    assert call_args.kwargs["params"] == {"on_conflict": "user_id,id"}
    row = call_args.kwargs["json_body"][0]
    assert row["user_id"] == saved_row["user_id"]
    assert row["expires_at"]


def test_supabase_get_scopes_by_user_and_conversation(
    monkeypatch: pytest.MonkeyPatch,
    supabase_environment: None,
) -> None:
    row = {
        "id": "conversation-1",
        "user_id": "4b36d83b-d58b-4db4-b215-5529ed4b2b2c",
        "title": "Requirements",
        "markdown": "",
        "messages_snapshot": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    request = MagicMock()
    request.return_value = request
    request.json.return_value = [row]
    monkeypatch.setattr(repository, "_supabase_request", request)

    result = repository.get_conversation(row["user_id"], row["id"])

    assert result is not None
    assert result.id == row["id"]
    assert request.call_args.kwargs["params"]["user_id"] == f"eq.{row['user_id']}"
