"""Tests for conversation repository: save, get, delete, list, update_drafts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import clear_settings_cache
from src.models.conversation import ConversationDocument
from src.services.conversation_repository import (
    delete_conversation,
    get_conversation,
    list_conversations,
    save_conversation,
    update_conversation_drafts,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def _mock_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set minimal env vars so get_settings() succeeds inside _ttl_seconds()."""
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret-key-must-be-32-chars!")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("REDIS_ENABLED", "false")
    monkeypatch.setenv("CHAT_SAVE_ENABLED", "false")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("DEFAULT_PUSH_TAG", "@AIGenerated")
    yield
    clear_settings_cache()


@pytest.fixture()
def mock_container() -> MagicMock:
    """Return a mock Cosmos container."""
    return MagicMock()


def _sample_cosmos_item(
    conversation_id: str = "conv-1",
    user_id: str = "user-1",
    title: str = "Test Conversation",
) -> dict[str, Any]:
    """Build a sample Cosmos item dict."""
    return {
        "id": conversation_id,
        "user_id": user_id,
        "title": title,
        "markdown": "# Test",
        "messages_snapshot": [{"role": "user", "content": "hi"}],
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }


# =============================================================================
# save_conversation Tests
# =============================================================================


class TestSaveConversation:
    """Tests for save_conversation create/update logic."""

    @patch("src.services.conversation_repository.get_conversations_container")
    def test_save_conversation_creates_new(
        self,
        mock_get_container: MagicMock,
        mock_container: MagicMock,
        _mock_env: None,
    ) -> None:
        """When no conversation_id, creates new doc with uuid."""
        mock_get_container.return_value = mock_container

        doc = save_conversation(
            "user-1",
            title="New Chat",
            markdown="Hello",
            messages_snapshot=[{"role": "user", "content": "hello"}],
        )

        assert doc.user_id == "user-1"
        assert doc.title == "New Chat"
        assert doc.id  # uuid was generated
        mock_container.upsert_item.assert_called_once()

    @patch("src.services.conversation_repository.get_conversation")
    @patch("src.services.conversation_repository.get_conversations_container")
    def test_save_conversation_updates_existing(
        self,
        mock_get_container: MagicMock,
        mock_get_conv: MagicMock,
        mock_container: MagicMock,
        _mock_env: None,
    ) -> None:
        """When conversation_id exists, updates it."""
        mock_get_container.return_value = mock_container

        existing = ConversationDocument(
            id="conv-exist",
            user_id="user-1",
            title="Old Title",
            markdown="Old markdown",
            messages_snapshot=[],
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        mock_get_conv.return_value = existing

        doc = save_conversation(
            "user-1",
            conversation_id="conv-exist",
            title="Updated Title",
            markdown="Updated markdown",
        )

        assert doc.id == "conv-exist"
        assert doc.title == "Updated Title"
        assert doc.created_at == existing.created_at
        mock_container.upsert_item.assert_called_once()

    @patch("src.services.conversation_repository.get_conversations_container")
    def test_save_conversation_no_container_raises(
        self,
        mock_get_container: MagicMock,
    ) -> None:
        """When container is None, raises RuntimeError."""
        mock_get_container.return_value = None

        with pytest.raises(RuntimeError, match="not configured"):
            save_conversation("user-1", title="Test")


# =============================================================================
# get_conversation Tests
# =============================================================================


class TestGetConversation:
    """Tests for get_conversation retrieval."""

    @patch("src.services.conversation_repository.get_conversations_container")
    def test_get_conversation_returns_doc(
        self,
        mock_get_container: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Mock container.read_item; verify ConversationDocument returned."""
        mock_get_container.return_value = mock_container
        mock_container.read_item.return_value = _sample_cosmos_item()

        result = get_conversation("user-1", "conv-1")

        assert result is not None
        assert isinstance(result, ConversationDocument)
        assert result.id == "conv-1"
        assert result.title == "Test Conversation"
        mock_container.read_item.assert_called_once_with(
            item="conv-1", partition_key="user-1"
        )

    @patch("src.services.conversation_repository.get_conversations_container")
    def test_get_conversation_not_found_returns_none(
        self,
        mock_get_container: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When read_item raises an exception, returns None."""
        mock_get_container.return_value = mock_container
        mock_container.read_item.side_effect = Exception("Not found")

        result = get_conversation("user-1", "nonexistent")

        assert result is None

    @patch("src.services.conversation_repository.get_conversations_container")
    def test_get_conversation_no_container_returns_none(
        self,
        mock_get_container: MagicMock,
    ) -> None:
        """When container is None, returns None without error."""
        mock_get_container.return_value = None

        result = get_conversation("user-1", "conv-1")

        assert result is None


# =============================================================================
# delete_conversation Tests
# =============================================================================


class TestDeleteConversation:
    """Tests for delete_conversation."""

    @patch("src.services.conversation_repository.get_conversation")
    @patch("src.services.conversation_repository.get_conversations_container")
    def test_delete_conversation_success(
        self,
        mock_get_container: MagicMock,
        mock_get_conv: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Mock container.delete_item; verify True returned."""
        mock_get_container.return_value = mock_container
        existing = ConversationDocument(
            id="conv-del",
            user_id="user-1",
            title="To Delete",
            markdown="",
            messages_snapshot=[],
        )
        mock_get_conv.return_value = existing

        result = delete_conversation("user-1", "conv-del")

        assert result is True
        mock_container.delete_item.assert_called_once_with(
            item="conv-del", partition_key="user-1"
        )

    @patch("src.services.conversation_repository.get_conversation")
    @patch("src.services.conversation_repository.get_conversations_container")
    def test_delete_conversation_not_found(
        self,
        mock_get_container: MagicMock,
        mock_get_conv: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When conversation doesn't exist, returns False."""
        mock_get_container.return_value = mock_container
        mock_get_conv.return_value = None

        result = delete_conversation("user-1", "nonexistent")

        assert result is False
        mock_container.delete_item.assert_not_called()

    @patch("src.services.conversation_repository.get_conversation")
    @patch("src.services.conversation_repository.get_conversations_container")
    def test_delete_conversation_exception_returns_false(
        self,
        mock_get_container: MagicMock,
        mock_get_conv: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When delete_item raises, returns False."""
        mock_get_container.return_value = mock_container
        existing = ConversationDocument(
            id="conv-err",
            user_id="user-1",
            title="Error",
            markdown="",
            messages_snapshot=[],
        )
        mock_get_conv.return_value = existing
        mock_container.delete_item.side_effect = Exception("Cosmos error")

        result = delete_conversation("user-1", "conv-err")

        assert result is False


# =============================================================================
# list_conversations Tests
# =============================================================================


class TestListConversations:
    """Tests for list_conversations."""

    @patch("src.services.conversation_repository.get_conversations_container")
    def test_list_conversations_returns_list(
        self,
        mock_get_container: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Mock container.query_items; verify list returned."""
        mock_get_container.return_value = mock_container
        mock_container.query_items.return_value = [
            {
                "id": "conv-1",
                "title": "Chat 1",
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-02T00:00:00+00:00",
            },
            {
                "id": "conv-2",
                "title": "Chat 2",
                "created_at": "2025-01-03T00:00:00+00:00",
                "updated_at": "2025-01-04T00:00:00+00:00",
            },
        ]

        result = list_conversations("user-1", limit=50)

        assert len(result) == 2
        assert all(isinstance(doc, ConversationDocument) for doc in result)
        assert result[0].id == "conv-1"
        assert result[1].id == "conv-2"

    @patch("src.services.conversation_repository.get_conversations_container")
    def test_list_conversations_with_days_filter(
        self,
        mock_get_container: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When days param is given, query includes cutoff filter."""
        mock_get_container.return_value = mock_container
        mock_container.query_items.return_value = []

        list_conversations("user-1", days=7)

        call_args = mock_container.query_items.call_args
        query = call_args.kwargs.get("query", call_args[1].get("query", ""))
        parameters = call_args.kwargs.get(
            "parameters", call_args[1].get("parameters", [])
        )

        assert "@cutoff" in query
        param_names = [p["name"] for p in parameters]
        assert "@cutoff" in param_names

    @patch("src.services.conversation_repository.get_conversations_container")
    def test_list_conversations_no_container_returns_empty(
        self,
        mock_get_container: MagicMock,
    ) -> None:
        """When container is None, returns empty list."""
        mock_get_container.return_value = None

        result = list_conversations("user-1")

        assert result == []


# =============================================================================
# update_conversation_drafts Tests
# =============================================================================


class TestUpdateConversationDrafts:
    """Tests for update_conversation_drafts."""

    @patch("src.services.conversation_repository.save_conversation")
    @patch("src.services.conversation_repository.get_conversation")
    def test_update_conversation_drafts_success(
        self,
        mock_get_conv: MagicMock,
        mock_save_conv: MagicMock,
    ) -> None:
        """Verify it calls save_conversation with drafts."""
        existing = ConversationDocument(
            id="conv-draft",
            user_id="user-1",
            session_id="sess-1",
            title="With Drafts",
            markdown="# Drafts",
            messages_snapshot=[],
        )
        mock_get_conv.return_value = existing
        mock_save_conv.return_value = existing

        drafts = [{"draft_id": "d1", "title": "Draft Item"}]
        result = update_conversation_drafts("user-1", "conv-draft", drafts)

        assert result is True
        mock_save_conv.assert_called_once()
        call_kwargs = mock_save_conv.call_args
        assert call_kwargs.kwargs["drafts_snapshot"] == drafts

    @patch("src.services.conversation_repository.get_conversation")
    def test_update_conversation_drafts_not_found(
        self,
        mock_get_conv: MagicMock,
    ) -> None:
        """When conversation not found, returns False."""
        mock_get_conv.return_value = None

        result = update_conversation_drafts("user-1", "nonexistent", [])

        assert result is False
